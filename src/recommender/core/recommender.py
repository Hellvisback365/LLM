"""
Sistema di raccomandazione multi-metrica basato su LangChain Agent.
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np # Aggiunto per la media delle metriche
import re
import sys  # Aggiunto per sys.stdout.flush()
import traceback # Aggiunto per debug
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from openai import RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Pydantic import
from pydantic import BaseModel, Field

# Moduli locali
from src.recommender.utils.data_processor import (
    load_ratings, 
    load_movies, 
    filter_users_by_specific_users,
    create_user_profiles
)
from src.recommender.utils.rag_utils import MovieRAG
from src.recommender.core.metrics_utils import calculate_precision_at_k, calculate_coverage

# ----------------------------
# Setup ambiente e parametri
# ----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in environment")

COMMON_LLM_PARAMS = {
    "openai_api_base": "https://openrouter.ai/api/v1",
    "openai_api_key": OPENROUTER_API_KEY,
    "temperature": 0.7,
    "max_tokens": 512, # Potrebbe essere necessario aumentarlo per l'evaluator
}
LLM_MODEL_ID = "mistralai/mistral-large-2411"

# Setup LLM con retry
llm = ChatOpenAI(model=LLM_MODEL_ID, **COMMON_LLM_PARAMS)

@retry(
    reraise=True,
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError)
)
async def llm_arun_with_retry(prompt_str: str) -> str:
    response = await llm.ainvoke(prompt_str)
    # Gestisce sia stringa che AIMessage
    content = getattr(response, 'content', None)
    return content if content is not None else str(response)

# ----------------------------
# Definizioni Schemi Pydantic (NUOVO)
# ----------------------------
class RecommendationOutput(BaseModel):
    """Schema per l'output dei tool di raccomandazione per metrica."""
    recommendations: List[int] = Field(..., description="Lista di esattamente 3 ID numerici di film raccomandati.")
    explanation: str = Field(..., description="Breve spiegazione testuale del motivo per cui questi film sono stati scelti in base alla metrica richiesta.")

class EvaluationOutput(BaseModel):
    """Schema per l'output del tool di valutazione finale."""
    final_recommendations: List[int] = Field(..., description="Lista finale OTTIMALE e UNICA di 3 ID numerici di film, bilanciando le metriche.")
    justification: str = Field(..., description="Spiegazione dettagliata della logica di selezione e bilanciamento per la lista finale aggregata.")
    trade_offs: str = Field(..., description="Descrizione dei trade-off considerati tra le diverse metriche (es. precisione vs copertura).")

# ----------------------------
# Definizioni Prompt e Parser (MODIFICATO)
# ----------------------------
PROMPT_VARIANTS = {
    "precision_at_k": (
        "Sei un sistema di raccomandazione esperto che ottimizza per PRECISION@K. "
        "Il tuo obiettivo è raccomandare film che l'utente valuterà con un rating 4 o 5 su 5. "
        "Analizza attentamente il profilo dell'utente e concentrati sui seguenti elementi:\n"
        "1. Generi che l'utente ha costantemente valutato con punteggi elevati\n"
        "2. Attributi specifici (attori, registi, temi, epoche) presenti nei film apprezzati\n"
        "3. Modelli nelle valutazioni negative per evitare film simili\n\n"
        "La precision@k misura quanti dei film raccomandati saranno effettivamente valutati positivamente. "
        "Quando analizzi il catalogo, presta particolare attenzione a:\n"
        "- Corrispondenza di genere con i film valutati positivamente\n"
        "- Somiglianza tematica e stilistica ai film preferiti\n"
        "- Evita film simili a quelli che l'utente non ha apprezzato\n\n"
        "NON raccomandare film in base alla popolarità generale o alle tendenze, a meno che queste "
        "caratteristiche non si allineino alle preferenze specifiche di questo utente. "
        "Fornisci ESATTAMENTE 3 film che l'utente molto probabilmente valuterà positivamente."
    ),
    "coverage": (
        "Sei un sistema di raccomandazione esperto che ottimizza per COVERAGE. "
        "Data una lista di film, consiglia 3 film che massimizzano la copertura di diversi generi cinematografici, "
        "MA che siano comunque rilevanti per le preferenze specifiche dell'utente di cui stai analizzando il profilo. "
        "La coverage misura la proporzione dell'intero catalogo che il sistema è in grado di raccomandare. "
        "L'obiettivo è esplorare meglio lo spazio dei film disponibili e ridurre il rischio di filter bubble. "
        "Assicurati che le tue raccomandazioni coprano generi diversi tra loro, ma che siano allineati con i gusti dell'utente. "
        "IMPORTANTE: Fai riferimento specifico ai film che l'utente ha apprezzato per scoprire generi correlati ma diversi. "
        "Ogni utente deve ricevere raccomandazioni personalizzate in base al suo profilo unico. "
        "NON raccomandare più di 3 film."
    )
}

def create_metric_prompt(metric_name: str, metric_description: str) -> PromptTemplate:
    """Crea un PromptTemplate per una specifica metrica (senza istruzioni formato JSON).""" # Modificato Docstring
    return PromptTemplate(
        input_variables=["catalog", "user_profile"],
        template=(
            f"# Compito: Raccomandatore di film ottimizzato per {metric_name.upper()}\n\n"
            f"{metric_description}\n\n"
            "# Profilo utente:\n"
            "{user_profile}\n\n"
            "# Catalogo film:\n"
            "{catalog}\n\n"
            "# Output Richiesto:\n"
            "Fornisci le raccomandazioni e la spiegazione richieste." # Semplificato, rimossi dettagli formato JSON
        )
    )

class RecommenderSystem:
    """
    Sistema di raccomandazione unificato basato su Agent e Tool.
    """
    
    def __init__(self, specific_user_ids: List[int] = [1, 2], model_id: str = LLM_MODEL_ID):
        self.specific_user_ids = specific_user_ids
        self.model_id = model_id
        self.llm = llm
        self.filtered_ratings = None
        self.user_profiles = None 
        self.movies = None
        self.rag = None
        self.agent = None
        self.metric_tools = []
        self.evaluator_tool = None
        self.datasets_loaded = False
        self.current_prompt_variants = PROMPT_VARIANTS.copy() # Inizializza con default

    def _load_datasets(self, force_reload: bool = False) -> None:
        """Carica e prepara i dataset internamente."""
        if not self.datasets_loaded or force_reload:
            print("\n=== Caricamento e processamento dei dataset ===")
            try:
                processed_dir = os.path.join('data', 'processed')
                os.makedirs(processed_dir, exist_ok=True)
                ratings_file = os.path.join(processed_dir, 'filtered_ratings_specific.csv')
                profiles_file = os.path.join(processed_dir, 'user_profiles.csv')
                movies_file = os.path.join(processed_dir, 'movies.csv')

                if not force_reload and all(os.path.exists(f) for f in [ratings_file, profiles_file, movies_file]):
                    print("Caricamento dati da file elaborati...")
                    self.filtered_ratings = pd.read_csv(ratings_file)
                    self.user_profiles = pd.read_csv(profiles_file, index_col=0) 
                    if self.user_profiles is not None:
                         self.user_profiles.index.name = 'user_id'
                    self.movies = pd.read_csv(movies_file)
                else:
                    print("Elaborazione dati dal dataset grezzo...")
                    ratings = load_ratings()
                    self.movies = load_movies()
                    self.filtered_ratings = filter_users_by_specific_users(ratings, self.specific_user_ids)
                    self.user_profiles = create_user_profiles(self.filtered_ratings)
                    self.filtered_ratings.to_csv(ratings_file, index=False)
                    self.user_profiles.to_csv(profiles_file, index=True) 
                    self.movies.to_csv(movies_file, index=False)
                
                if self.user_profiles is None or self.user_profiles.empty:
                     print("ATTENZIONE: user_profiles vuoto dopo il caricamento!")
                
                print(f"Dataset processato: {len(self.movies)} film, {len(self.user_profiles) if self.user_profiles is not None else 0} profili (ID: {self.specific_user_ids}).")
                self.datasets_loaded = True
            except Exception as e:
                print(f"Errore caricamento dataset: {e}")
                traceback.print_exc()
                raise
        else:
            print("Dataset già caricati.")

    def _initialize_rag(self, force_recreate_vector_store: bool = False) -> None:
        """Inizializza il sistema RAG."""
        if not self.datasets_loaded:
            self._load_datasets()
        if self.movies is None or self.movies.empty:
             print("Movies non caricati, impossibile inizializzare RAG."); return
        print("\n=== Preparazione RAG ===")
        self.rag = MovieRAG()
        movies_list = self.movies.to_dict('records')
        self.rag.load_or_create_vector_store(movies_list, force_recreate=force_recreate_vector_store)
        try:
            catalog_json = self.rag.get_optimized_catalog_for_llm(movies_list)
            catalog_path = os.path.join('data', 'processed', 'optimized_catalog.json')
            os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
            with open(catalog_path, 'w', encoding='utf-8') as f: f.write(catalog_json)
            print(f"Catalogo ottimizzato RAG salvato.")
        except Exception as e: print(f"Attenzione: impossibile generare/salvare catalogo ottimizzato RAG: {e}")

    def _build_metric_tools(self) -> List[Tool]:
        """Costruisce i Tools per ciascuna metrica di raccomandazione."""
        print("Costruzione metric tools...")
        metric_tools = []
        prompt_variants_to_use = self.current_prompt_variants # Usa le varianti correnti

        for metric_name, metric_desc in prompt_variants_to_use.items():
            prompt_template = create_metric_prompt(metric_name, metric_desc)
            
            async def run_metric_agent_tool(catalog: str, user_profile: str, _metric=metric_name, _prompt=prompt_template):
                try:
                    prompt_str = _prompt.format(catalog=catalog, user_profile=user_profile)

                    # Crea un'istanza LLM strutturata al volo per questa chiamata (MODIFICATO)
                    # Rimuovi method=\"json_mode\" per usare il tool calling (default)
                    structured_llm = self.llm.with_structured_output(RecommendationOutput)
                    # Potremmo forzare l'uso del tool se necessario:
                    # structured_llm = self.llm.with_structured_output(RecommendationOutput).bind(tool_choice=\"RecommendationOutput\")

                    print(f"Invoking structured LLM for metric: {_metric} (using tool calling approach)") # Log aggiornato
                    parsed_response: RecommendationOutput = await structured_llm.ainvoke(prompt_str)
                    print(f"Raw structured response from {_metric}: {parsed_response}") # Log

                    # Restituisci il risultato come dizionario
                    return {"metric": _metric, **parsed_response.dict()}

                except Exception as e:
                    print(f"Error running structured {_metric} tool: {e}")
                    traceback.print_exc() # Stampa traceback completo
                    # Fornisci un output di errore strutturato coerente
                    return {"metric": _metric, "recommendations": [], "explanation": f"Execution Error: {e}"}
            
            metric_tools.append(
                Tool(
                    name=f"recommender_{metric_name}",
                    func=None,
                    description=f"Genera raccomandazioni ottimizzate per {metric_name}. Input: catalog (JSON string), user_profile (JSON string)",
                    coroutine=run_metric_agent_tool
                )
            )
        self.metric_tools = metric_tools
        return metric_tools

    def _build_evaluator_tool(self) -> Tool:
        """Costruisce il Tool per valutare e combinare le raccomandazioni."""
        print("Costruzione evaluator tool...")
        eval_prompt = PromptTemplate(
            input_variables=["all_recommendations", "catalog"],
             template=(
                 "# Compito: Valutazione e combinazione di raccomandazioni multi-metrica\n\n"
                 "Sei un sistema avanzato che deve combinare raccomandazioni di film generate da diversi sistemi specializzati.\n"
                 "Ogni sistema si è concentrato su una metrica diversa (es. precision@k, coverage).\n\n"
                 "# Raccomandazioni Ricevute per Utente (JSON string):\n"
                 "{all_recommendations}\n\n"
                 "# Catalogo Film (JSON string):\n"
                 "{catalog}\n\n"
                 "# Istruzioni:\n"
                 "1. Analizza attentamente le raccomandazioni per ogni metrica e per ogni utente.\n"
                 "2. Considera i punti di forza di ciascuna metrica (precision@k = rilevanza, coverage = diversità).\n"
                 "3. Crea una lista finale OTTIMALE e UNICA di 3 film che bilanci le diverse metriche in modo sensato per un ipotetico utente medio rappresentato dai dati forniti.\n"
                 "4. Spiega DETTAGLIATAMENTE la tua logica di selezione e i trade-off considerati per arrivare alla lista finale aggregata.\n\n"
                 "# Output Richiesto:\n"
                 "Fornisci le raccomandazioni finali, la giustificazione e i trade-off." # Semplificato, rimossi dettagli formato JSON
             )
        )
        
        async def evaluate_recommendations_tool(all_recommendations_str: str, catalog_str: str) -> Dict:
            max_retries = 3
            retry_delay = 2
            evaluator_max_tokens = 1536 # NUOVO: Aumenta i token per l'evaluator

            for attempt in range(max_retries):
                try:
                    print(f"\nAttempt {attempt+1}/{max_retries} to evaluate recommendations using structured output (max_tokens={evaluator_max_tokens})...") # Log aggiornato
                    prompt_str = eval_prompt.format(all_recommendations=all_recommendations_str, catalog=catalog_str)

                    # Crea l'istanza LLM strutturata per l'evaluator (MODIFICATO)
                    # Rimuovi method="json_mode" per usare il tool calling (default)
                    structured_eval_llm = self.llm.with_structured_output(EvaluationOutput)

                    # Passa max_tokens maggiorato
                    parsed_evaluation: EvaluationOutput = await structured_eval_llm.ainvoke(
                        prompt_str,
                        max_tokens=evaluator_max_tokens
                    )
                    print(f"Raw structured evaluator response: {parsed_evaluation}") # Log

                    # L'output è già validato da Pydantic e LangChain. Restituiscilo come dict.
                    return parsed_evaluation.dict()

                except Exception as inner_e:
                    print(f"Error during structured evaluator tool execution attempt {attempt+1}: {inner_e}")
                    traceback.print_exc() # Stampa traceback completo
                    if "LengthFinishReasonError" in str(inner_e) and evaluator_max_tokens < 3000: # Aumenta se ancora troppo corto
                         print("Increasing max_tokens for evaluator retry...")
                         evaluator_max_tokens *= 1.5 # Aumenta gradualmente
                         evaluator_max_tokens = int(evaluator_max_tokens)

                    if attempt < max_retries - 1:
                         await asyncio.sleep(retry_delay); retry_delay *= 2
                    else:
                        # Restituisci un errore strutturato coerente
                        return {"final_recommendations": [], "justification": f"Evaluation failed after retries: {inner_e}", "trade_offs": "N/A"}
            # In caso di uscita imprevista dal loop
            return {"final_recommendations": [], "justification": "Evaluation failed unexpectedly.", "trade_offs": "N/A"}

        self.evaluator_tool = Tool(
            name="evaluate_recommendations",
            func=None,
            description="Valuta e combina raccomandazioni generate. Input: all_recommendations (JSON string), catalog (JSON string)",
            coroutine=evaluate_recommendations_tool
        )
        return self.evaluator_tool
        
    def _initialize_agent(self) -> None:
        """Inizializza l'Agent di LangChain con i tool costruiti."""
        print("Inizializzazione Agent...")
        if not self.metric_tools: self._build_metric_tools()
        if not self.evaluator_tool: self._build_evaluator_tool()
        all_tools = self.metric_tools + [self.evaluator_tool]
        try:
            if not all(hasattr(tool, 'coroutine') and callable(tool.coroutine) for tool in all_tools):
                raise ValueError("Tutti i tools devono avere una coroutine.")
            self.agent = initialize_agent(
                all_tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                verbose=True, handle_parsing_errors=True
            )
            print("LangChain Agent inizializzato.")
        except Exception as e:
             print(f"Errore inizializzazione Agent: {e}")
             traceback.print_exc()
             self.agent = None

    def initialize_system(self, force_reload_data: bool = False, force_recreate_vector_store: bool = False) -> None:
        """Metodo pubblico per inizializzare o reinizializzare il sistema."""
        print("\n=== Inizializzazione Sistema ===")
        self._load_datasets(force_reload=force_reload_data)
        self._initialize_rag(force_recreate_vector_store=force_recreate_vector_store)
        self._initialize_agent()
        print("=== Sistema Inizializzato ===")

    def get_optimized_catalog(self, limit: int = 100) -> str:
        """Ottiene il catalogo ottimizzato per l'LLM."""
        catalog_path = os.path.join('data', 'processed', 'optimized_catalog.json')
        try:
            if os.path.exists(catalog_path):
                with open(catalog_path, 'r', encoding='utf-8') as f: catalog_data = json.load(f)
                return json.dumps(catalog_data[:limit] if limit else catalog_data, ensure_ascii=False)
            elif self.rag:
                 print("Catalogo ottimizzato non trovato, genero da RAG...")
                 if self.movies is None or self.movies.empty: self._load_datasets()
                 movies_list = self.movies.to_dict('records')
                 return self.rag.get_optimized_catalog_for_llm(movies_list, limit=limit)
            else: print("Attenzione: RAG non inizializzato."); return "[]"
        except Exception as e: print(f"Errore get_optimized_catalog: {e}"); return "[]"
             
    async def run_recommendation_pipeline(self, use_prompt_variants: Dict = None) -> Tuple[Dict, Dict, Dict[int, List[int]]]:
        """
        Esegue l'intera pipeline di raccomandazione per tutti gli utenti specificati.
        Invoca i tool delle metriche e poi il tool di valutazione.
        """
        if not self.agent: raise RuntimeError("Agent non inizializzato. Chiamare initialize_system() prima.")
        if not self.datasets_loaded or self.user_profiles is None or self.user_profiles.empty:
             raise RuntimeError("Dataset non caricati o profili utente vuoti. Chiamare initialize_system() prima.")

        # Imposta le varianti di prompt da usare per questa run
        self.current_prompt_variants = use_prompt_variants if use_prompt_variants is not None else PROMPT_VARIANTS
        # Ricostruisce i tool metriche con le varianti correnti
        current_metric_tools = self._build_metric_tools() 
        # Nota: l'agent NON viene reinizializzato qui, usa i tool passati al momento della chiamata (se usassimo agent.arun)

        user_metric_results = {}
        per_user_held_out_items = {} # NUOVO: Dizionario per item hold-out per utente
        if self.user_profiles is None: raise RuntimeError("user_profiles non inizializzato.")

        for user_id in self.specific_user_ids:
            if user_id not in self.user_profiles.index:
                print(f"Attenzione: Utente {user_id} non trovato."); continue

            profile_series = self.user_profiles.loc[user_id]
            def safe_load_list(item):
                 if isinstance(item, list): return item
                 if pd.isna(item): return []
                 if isinstance(item, str):
                     try: return json.loads(item.replace("'", '"')) if isinstance(json.loads(item.replace("'", '"')), list) else []
                     except: 
                         if item.startswith('[') and item.endswith(']'): 
                             try: return [int(x.strip()) for x in item[1:-1].split(',') if x.strip().isdigit()]
                             except: pass
                 return []
            profile_liked = safe_load_list(profile_series.get("profile_liked_movies")) 
            disliked = safe_load_list(profile_series.get("disliked_movies"))
            profile_summary = json.dumps({"user_id": int(user_id),"liked_movies": profile_liked,"disliked_movies": disliked}, ensure_ascii=False)

            # NUOVO: Colleziona gli item held-out per questo utente
            held_out = safe_load_list(profile_series.get("held_out_liked_movies"))
            per_user_held_out_items[user_id] = held_out # Salva nel dizionario

            print(f"\n--- Raccomandazioni per utente {user_id} --- (Profilo con {len(profile_liked)} liked, {len(disliked)} disliked. Held-out: {len(held_out)} items)") # Log aggiornato
            
            # Fix Linter Indentation Error
            catalog_json = "[]"
            try:
                 if self.rag:
                     cat_p = self.rag.similarity_search(profile_summary, k=100, metric_focus="precision_at_k", user_id=int(user_id))
                     cat_c = self.rag.similarity_search("diversi generi " + profile_summary, k=100, metric_focus="coverage", user_id=int(user_id))
                     merged = self.rag.merge_catalogs(cat_p, cat_c)
                     catalog_json = json.dumps(merged[:100], ensure_ascii=False)
                 else: catalog_json = self.get_optimized_catalog(limit=100)
            except Exception as e: print(f"Errore RAG user {user_id}: {e}. Uso catalogo generico."); catalog_json = self.get_optimized_catalog(limit=100)
            
            # Esegui i tool delle metriche direttamente
            results_for_user = {}
            tasks = [tool.coroutine(catalog=catalog_json, user_profile=profile_summary) for tool in current_metric_tools if tool.coroutine]
            tool_names = [tool.name.replace("recommender_", "") for tool in current_metric_tools if tool.coroutine]
            
            if tasks:
                metric_outputs = await asyncio.gather(*tasks, return_exceptions=True)
                for i, output in enumerate(metric_outputs):
                    metric_name = tool_names[i]
                    if isinstance(output, Exception): results_for_user[metric_name] = {"metric": metric_name, "recommendations": [], "explanation": f"Error: {output}"}
                    elif isinstance(output, dict): results_for_user[metric_name] = output
                    else: results_for_user[metric_name] = {"metric": metric_name, "recommendations": [], "explanation": f"Unexpected type: {type(output)}"}
            user_metric_results[user_id] = results_for_user

        # Valutazione aggregata
        print("\n--- Valutazione Aggregata ---")
        final_evaluation = {}
        if user_metric_results:
            try:
                all_results_str = json.dumps(user_metric_results, ensure_ascii=False, indent=2)
                eval_catalog = self.get_optimized_catalog(limit=100)
                if self.evaluator_tool and self.evaluator_tool.coroutine:
                    final_evaluation = await self.evaluator_tool.coroutine(all_recommendations_str=all_results_str, catalog_str=eval_catalog)
                else: raise ValueError("Evaluator tool/coroutine non definito.")
                print(f"Final recommendations: {final_evaluation.get('final_recommendations', [])}")
            except Exception as e:
                 print(f"Errore evaluator tool: {e}"); traceback.print_exc()
                 final_evaluation = {"final_recommendations": [], "justification": f"Evaluation Error: {e}", "trade_offs": "N/A"}
        else: final_evaluation = {"final_recommendations": [], "justification": "No results to evaluate.", "trade_offs": "N/A"}

        # Ripristina le varianti di prompt di default se necessario
        self.current_prompt_variants = PROMPT_VARIANTS.copy() 
        self._build_metric_tools() # Ricostruisce i tool con i prompt di default

        # Passa per_user_held_out_items alla funzione di calcolo metriche
        return user_metric_results, final_evaluation, per_user_held_out_items # Restituisce dizionario hold-out

    def save_results(self, metric_results: Dict, final_evaluation: Dict, metrics_calculated: Dict = None, per_user_held_out_items: Dict[int, List[int]] = None):
        """Salva i risultati della pipeline su file."""
        # Usa un blocco per garantire che il file sia chiuso prima di procedere
        try:
            with open("metric_recommendations.json", "w", encoding="utf-8") as f:
                json.dump(metric_results, f, ensure_ascii=False, indent=2)
                # Flush esplicito per garantire che il file sia scritto completamente
                f.flush()
                os.fsync(f.fileno())
            print("\nRisultati intermedi salvati: metric_recommendations.json")
            sys.stdout.flush()  # Flush dello stdout
        except Exception as e: 
            print(f"Errore salvataggio metric_recommendations.json: {e}")
            sys.stdout.flush()
        
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "metric_recommendations": metric_results,
            "final_evaluation": final_evaluation
        }
        if metrics_calculated: 
            result_data["metrics"] = metrics_calculated
        if per_user_held_out_items is not None: # NUOVO: Aggiungi item hold-out per utente al salvataggio
             # Converte le chiavi user_id (int) in stringhe per compatibilità JSON
             result_data["per_user_held_out_items"] = {str(k): v for k, v in per_user_held_out_items.items()}
        
        try:
            with open("recommendation_results.json", "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                # Flush esplicito per garantire che il file sia scritto completamente
                f.flush()
                os.fsync(f.fileno())
            print("Risultati finali salvati: recommendation_results.json")
            sys.stdout.flush()  # Flush dello stdout
        except Exception as e: 
            print(f"Errore salvataggio recommendation_results.json: {e}")
            sys.stdout.flush()

    def calculate_and_display_metrics(self, metric_results: Dict, final_evaluation: Dict, per_user_relevant_items: Dict[int, List[int]]) -> Dict:
        """Calcola e visualizza metriche per utente e aggregate.
        
        Args:
            metric_results: Dizionario {user_id: {metric_name: results}}
            final_evaluation: Dizionario con le raccomandazioni finali aggregate.
            per_user_relevant_items: Dizionario {user_id: list_of_held_out_ids}
            
        Returns:
            Dizionario con metriche per utente e metriche aggregate (medie).
        """
        if not self.datasets_loaded or self.movies is None:
             print("Dataset non caricati, metriche non calcolabili."); return {}
        
        per_user_metrics = {}
        all_final_recs = final_evaluation.get('final_recommendations', []) # Raccomandazioni finali aggregate
        # Calcola precision@k per le raccomandazioni finali vs *tutti* gli item hold-out aggregati
        # (Manteniamo questa metrica aggregata per le final recs, dato che sono aggregate)
        all_relevant_items_flat = [item for sublist in per_user_relevant_items.values() for item in sublist]
        final_pak_value_agg = calculate_precision_at_k(all_final_recs, all_relevant_items_flat)
        
        # Metriche per utente
        metric_names = list(self.current_prompt_variants.keys()) # Ottiene nomi metriche (es. precision_at_k, coverage)
        aggregated_metrics = {name: {'precision_scores': [], 'genre_coverage_scores': []} for name in metric_names}
        aggregated_metrics['final'] = {'precision_scores': [final_pak_value_agg], 'genre_coverage_scores': []} # Aggiunge metrica finale aggregata

        print("\nMetriche Calcolate (Per Utente):")
        for user_id, u_metrics in metric_results.items():
            user_relevant = per_user_relevant_items.get(user_id, [])
            if not user_relevant:
                print(f"  Utente {user_id}: Attenzione - Nessun item rilevante (held-out) trovato.")
                # Continua a calcolare le altre metriche, la precisione sarà 0
                
            user_metrics_calculated = {}
            print(f"  Utente {user_id}:")
            
            # Calcola metriche per i tool (precision_at_k, coverage, ecc.)
            for metric_name in metric_names:
                metric_data = u_metrics.get(metric_name, {})
                recs = metric_data.get('recommendations', [])
                pak_value = calculate_precision_at_k(recs, user_relevant)
                
                # Genre Coverage (uguale a prima ma calcolata qui)
                genres = set() 
                if self.movies is not None:
                    def get_genres(mid): 
                        m = self.movies[self.movies['movie_id'] == mid]
                        return set(m.iloc[0]['genres'].split('|')) if not m.empty and pd.notna(m.iloc[0]['genres']) else set()
                    genres = set().union(*[get_genres(mid) for mid in recs])
                    all_available_genres = set(g for movie_genres in self.movies['genres'].dropna() for g in movie_genres.split('|'))
                    n_genres = len(all_available_genres)
                    genre_cov = len(genres) / n_genres if n_genres > 0 else 0.0
                else:
                    genre_cov = 0.0

                user_metrics_calculated[metric_name] = {
                    "precision_score": pak_value,
                    "genre_coverage": genre_cov
                }
                aggregated_metrics[metric_name]['precision_scores'].append(pak_value)
                aggregated_metrics[metric_name]['genre_coverage_scores'].append(genre_cov)
                print(f"    {metric_name}: Precision@k={pak_value:.4f}, GenreCoverage={genre_cov:.4f}")

            per_user_metrics[user_id] = user_metrics_calculated

        # Calcola medie aggregate (MAP@k e Mean Genre Coverage)
        print("\nMetriche Aggregate (Medie su Utenti):")
        final_metrics_summary = {"per_user": per_user_metrics, "aggregate_mean": {}}
        
        # Calcola Genre Coverage per le raccomandazioni finali aggregate
        final_genres = set() 
        if self.movies is not None:
            def get_genres_final(mid): 
                m = self.movies[self.movies['movie_id'] == mid]
                return set(m.iloc[0]['genres'].split('|')) if not m.empty and pd.notna(m.iloc[0]['genres']) else set()
            final_genres = set().union(*[get_genres_final(mid) for mid in all_final_recs])
            all_available_genres = set(g for movie_genres in self.movies['genres'].dropna() for g in movie_genres.split('|'))
            n_genres = len(all_available_genres)
            final_genre_cov_agg = len(final_genres) / n_genres if n_genres > 0 else 0.0
        else:
            final_genre_cov_agg = 0.0
        aggregated_metrics['final']['genre_coverage_scores'].append(final_genre_cov_agg) # Aggiungi per il report finale

        for name in metric_names + ['final']: # Include 'final' nel loop
            avg_pak = np.mean(aggregated_metrics[name]['precision_scores']) if aggregated_metrics[name]['precision_scores'] else 0.0
            avg_gen_cov = np.mean(aggregated_metrics[name]['genre_coverage_scores']) if aggregated_metrics[name]['genre_coverage_scores'] else 0.0
            label = f"Mean {name.capitalize()}" if name != 'final' else "Final Aggregated"
            
            # Per 'final', usiamo i valori aggregati calcolati (pak vs all, genre cov vs all)
            if name == 'final':
                 print(f"  {label}: Precision@k (vs all held-out)={final_pak_value_agg:.4f}, GenreCoverage={final_genre_cov_agg:.4f}")
                 final_metrics_summary["aggregate_mean"][name] = {"precision_score_agg": final_pak_value_agg, "genre_coverage": final_genre_cov_agg}
            else:
                 print(f"  {label}: MAP@k={avg_pak:.4f}, Mean GenreCoverage={avg_gen_cov:.4f}")
                 final_metrics_summary["aggregate_mean"][name] = {"map_at_k": avg_pak, "mean_genre_coverage": avg_gen_cov}
        
        # Calcolo Total Item Coverage (come prima, aggregato)
        all_recs_flat = []
        for uid, u_metrics in metric_results.items():
             for m_name, m_data in u_metrics.items():
                  all_recs_flat.extend(m_data.get('recommendations', []))
        all_recs_flat.extend(all_final_recs)
        total_item_coverage = len(set(all_recs_flat)) / len(self.movies['movie_id'].unique()) if self.movies is not None and not self.movies.empty else 0.0
        print(f"  Total Item Coverage (all recs): {total_item_coverage:.4f}")
        final_metrics_summary["aggregate_mean"]["total_item_coverage"] = total_item_coverage

        return final_metrics_summary
        
    # ----- Metodi per esperimenti ----- 
    async def generate_recommendations_with_custom_prompt(self, prompt_variants: Dict, experiment_name: str ="custom_experiment") -> Tuple[Dict, str]:
        print(f"\n=== Esecuzione Esperimento: {experiment_name} ===")
        # MODIFICA: Recupera per_user_held_out_items
        metric_results, final_evaluation, per_user_held_out_items = await self.run_recommendation_pipeline(use_prompt_variants=prompt_variants)
        # MODIFICA: Passa per_user_held_out_items
        metrics = self.calculate_and_display_metrics(metric_results, final_evaluation, per_user_held_out_items)
        os.makedirs("experiments", exist_ok=True)
        filename = f"experiments/experiment_{experiment_name}.json"
        result = {
            "timestamp": datetime.now().isoformat(),
            "experiment_info": {"name": experiment_name, "prompt_variants": prompt_variants},
            "metric_recommendations": metric_results,
            "final_evaluation": final_evaluation,
            "metrics": metrics, # Ora contiene la struttura per-utente e aggregata
            "per_user_held_out_items": {str(k): v for k, v in per_user_held_out_items.items()} # Aggiungi dizionario hold-out
        }
        try:
            with open(filename, "w", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Risultati esperimento salvati: {filename}")
        except Exception as e: print(f"Errore salvataggio file esperimento {filename}: {e}")
        return result, filename
        
    async def generate_standard_recommendations(self) -> Dict:
        """Genera raccomandazioni standard, calcola metriche e salva."""
        print("\n=== Esecuzione Pipeline Standard ===")
        # MODIFICA: Recupera per_user_held_out_items
        metric_results, final_evaluation, per_user_held_out_items = await self.run_recommendation_pipeline()
        # MODIFICA: Passa per_user_held_out_items
        metrics = self.calculate_and_display_metrics(metric_results, final_evaluation, per_user_held_out_items)
        # MODIFICA: Passa per_user_held_out_items
        self.save_results(metric_results, final_evaluation, metrics_calculated=metrics, per_user_held_out_items=per_user_held_out_items)
        
        # Dà tempo all'event loop di stabilizzarsi prima di stampare i messaggi finali
        await asyncio.sleep(0.1)
        sys.stdout.flush()
        
        print("\n=== Standard Recommendation Process Complete ===")
        print(f"Final recommendations: {final_evaluation.get('final_recommendations', [])}")
        sys.stdout.flush()
        
        return {"timestamp": datetime.now().isoformat(), "metric_recommendations": metric_results, "final_evaluation": final_evaluation, "metrics": metrics} # Restituisce nuove metriche 