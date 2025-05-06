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
import time

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
    "temperature": 0.4,
    "max_tokens": 1536, # AUMENTATO da 512 per permettere output più lunghi (50 recs + spiegazione)
}


LLM_MODEL_ID = "meta-llama/llama-3.3-70b-instruct"

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
    recommendations: List[int] = Field(..., description="Lista ORDINATA di fino a 50 ID numerici di film raccomandati. Il primo ID è il più raccomandato, l'ultimo il meno.")
    explanation: str = Field(..., description="Breve spiegazione testuale del motivo per cui questi film sono stati scelti e ordinati in base alla metrica richiesta.")

class EvaluationOutput(BaseModel):
    """Schema per l'output del tool di valutazione finale."""
    final_recommendations: List[int] = Field(..., description="Lista finale OTTIMALE e ORDINATA di fino a 50 ID numerici di film, bilanciando le metriche. Il primo ID è il più raccomandato.")
    justification: str = Field(..., description="Spiegazione dettagliata della logica di selezione, bilanciamento e ORDINAMENTO per la lista finale aggregata.")
    trade_offs: str = Field(..., description="Descrizione dei trade-off considerati tra le diverse metriche (es. precisione vs copertura) nell'ordinamento finale.")

# ----------------------------
# Definizioni Prompt e Parser (MODIFICATO)
# ----------------------------
NUM_RECOMMENDATIONS = 50 # NUOVA costante per il numero di raccomandazioni

PROMPT_VARIANTS = {
    "precision_at_k": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a personal movie recommendation consultant optimizing for PRECISION@K for a specific user. "
        "Your goal is to recommend movies that the user will rate 4 or 5 out of 5. "
        "Carefully analyze the user\'s profile and focus on the following elements:\n"
        "1. Genres that the user has consistently rated highly.\n"
        "2. Identify key actors, directors, themes, and time periods from the user\'s appreciated movies. Prioritize movies in the catalog that share these specific attributes.\n"
        "3. Actively analyze disliked movies to identify genres, themes, or attributes to avoid.\n\n"
        "Precision@k measures how many of the recommended movies will actually be rated positively. "
        "When analyzing the catalog, pay particular attention to:\n"
        "- Genre matching with positively rated movies.\n"
        "- Thematic and stylistic similarity to favorite movies.\n"
        "- Avoid movies similar to those the user did not appreciate.\n\n"
        "DO NOT recommend movies based on general popularity or trends, unless these "
        "characteristics align with this specific user\'s unique preferences. "
        f"Provide an ORDERED list of {NUM_RECOMMENDATIONS} movie IDs. The first movie should be the one you recommend the most, "
        "the last one the one you recommend the least, based on the probability that the user will rate them positively. "
        "Your explanation should briefly outline the main reasons for your top selections in relation to the user\'s profile."
        "<|eot_id|>" # Note: The user/assistant turns will be added dynamically when the prompt is used.
    ),
    "coverage": (
        f"You are an expert recommendation system that optimizes for COVERAGE. "
        f"Given a list of movies, recommend an ORDERED list of {NUM_RECOMMENDATIONS} movies that maximize coverage of different film genres, "
        f"BUT that are still relevant to the specific preferences of the user whose profile you are analyzing. "
        f"Coverage measures the proportion of the entire catalog that the system is able to recommend. "
        f"The goal is to better explore the available movie space and reduce the risk of filter bubbles. "
        f"Make sure your recommendations cover different genres, but are aligned with the user's tastes. "
        f"Order the list by putting first the movies that represent a good compromise between genre diversity and user preferences, "
        f"and last those that prioritize pure diversity more at the expense of immediate relevance. "
        f"IMPORTANT: Make specific reference to movies the user has enjoyed to discover related but different genres. "
        f"Each user should receive personalized recommendations based on their unique profile. "
        f"DO NOT recommend more than {NUM_RECOMMENDATIONS} movies."
    )
}

def create_metric_prompt(metric_name: str, metric_description: str) -> PromptTemplate:
    """Crea un PromptTemplate per una specifica metrica (senza istruzioni formato JSON).""" # Modificato Docstring
    return PromptTemplate(
        input_variables=["catalog", "user_profile"],
        template=(
            f"# Task: Movie recommender optimized for {metric_name.upper()}\n\n"
            f"{metric_description}\n\n"
            "# User profile:\n"
            "{user_profile}\n\n"
            "# Movie catalog:\n"
            "{catalog}\n\n"
            "# Required Output:\n"
            "Provide the recommendations and explanation requested." # Semplificato, rimossi dettagli formato JSON
        )
    )

class RecommenderSystem:
    """
    Sistema di raccomandazione unificato basato su Agent e Tool.
    """
    
    def __init__(self, specific_user_ids: List[int] = [4277, 4169, 1680], model_id: str = LLM_MODEL_ID):
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

                    # Seleziona il metodo per l'output strutturato
                    # Cambia questa variabile per testare diversi metodi:
                    # "function_calling", "json_schema", o "json_mode"
                    structured_output_method = "function_calling"
                    # structured_output_method = "json_schema"

                    structured_llm = self.llm.with_structured_output(
                        RecommendationOutput, 
                        method=structured_output_method
                    )
                    # Nota: se si usa "json_mode", è necessario aggiungere format_instructions al prompt.

                    print(f"Invoking structured LLM for metric: {_metric} (using method: {structured_output_method})") # Log aggiornato
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
                 "# Task: Evaluation and combination of multi-metric recommendations\n\n"
                 "You are an advanced system that must combine movie recommendations generated by different specialized systems.\n"
                 "Each system has focused on a different metric (e.g., precision@k, coverage).\n\n"
                 "# Recommendations Received per User (JSON string):\n"
                 "{all_recommendations}\n\n"
                 "# Movie Catalog (JSON string):\n"
                 "{catalog}\n\n"
                 "# Instructions:\n"
                 "1. Carefully analyze the recommendations for each metric and for each user.\n"
                 "2. Consider the strengths of each metric (precision@k = relevance, coverage = diversity).\n"
                 f"3. Create a final OPTIMAL and ORDERED list of {NUM_RECOMMENDATIONS} movies that sensibly balances the different metrics for a hypothetical average user represented by the provided data. Order the list from the most recommended movie to the least recommended.\n"
                 "4. THOROUGHLY explain your selection logic, the trade-offs considered, and how you determined the ORDERING to arrive at the final aggregated list.\n\n"
                 "# Required Output:\n"
                 "Provide the ordered final recommendations, justification, and trade-offs." # Semplificato, rimossi dettagli formato JSON
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

                    # Seleziona il metodo per l'output strutturato per l'evaluator
                    # Cambia questa variabile per testare diversi metodi:
                    evaluator_structured_output_method = "function_calling"
                    # evaluator_structured_output_method = "json_schema"

                    structured_eval_llm = self.llm.with_structured_output(
                        EvaluationOutput, 
                        method=evaluator_structured_output_method
                    )
                    # Nota: se si usa "json_mode", è necessario aggiungere format_instructions al prompt.

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

        start_all_users = time.time()
        for user_id in self.specific_user_ids:
            start_user = time.time()
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
            
            # NUOVO: Genera cataloghi specifici per metrica
            metric_specific_catalogs = {}
            catalog_json_fallback = self.get_optimized_catalog(limit=300) # Catalogo di fallback generico

            if self.rag:
                try:
                    # Catalogo per Precision@k
                    print(f"RAG: Tentativo chiamata similarity_search per precision_at_k per utente {user_id}...") # LOG AGGIUNTO
                    start_rag_p = time.time()
                    cat_p = self.rag.similarity_search(profile_summary, k=300, metric_focus="precision_at_k", user_id=int(user_id))
                    end_rag_p = time.time()
                    print(f"RAG: Tempo impiegato per precision_at_k: {end_rag_p - start_rag_p:.2f} secondi")
                    metric_specific_catalogs["precision_at_k"] = json.dumps(cat_p[:300], ensure_ascii=False)
                    print(f"RAG: Generato catalogo specifico per precision_at_k (size: {len(cat_p)}) (Successo)") # Log modificato per chiarezza
                except Exception as e:
                    print(f"Errore RAG (precision_at_k) user {user_id}: {e}. Uso catalogo fallback.")
                    metric_specific_catalogs["precision_at_k"] = catalog_json_fallback

                try:
                    # Catalogo per Coverage
                    # Usiamo una query diversa per coverage per enfatizzare la diversità
                    coverage_query = "diversi generi film non ancora visti dall'utente" + profile_summary 
                    print(f"RAG: Tentativo chiamata similarity_search per coverage per utente {user_id}...") # LOG AGGIUNTO
                    start_rag_c = time.time()
                    cat_c = self.rag.similarity_search(coverage_query, k=300, metric_focus="coverage", user_id=int(user_id))
                    end_rag_c = time.time()
                    print(f"RAG: Tempo impiegato per coverage: {end_rag_c - start_rag_c:.2f} secondi")
                    metric_specific_catalogs["coverage"] = json.dumps(cat_c[:300], ensure_ascii=False)
                    print(f"RAG: Generato catalogo specifico per coverage (size: {len(cat_c)}) (Successo)") # Log modificato per chiarezza
                except Exception as e:
                    print(f"Errore RAG (coverage) user {user_id}: {e}. Uso catalogo fallback.")
                    metric_specific_catalogs["coverage"] = catalog_json_fallback
            else:
                # Se RAG non è disponibile, usa il fallback per tutte le metriche
                print("RAG non disponibile. Uso catalogo fallback per tutte le metriche.")
                # Potremmo popolare metric_specific_catalogs qui per metriche note
                # se volessimo che il fallback fosse comunque specifico per metrica
                # Ma per ora, lasceremo che il get successivo usi catalog_json_fallback
                pass # Il fallback verrà gestito nel loop sottostante

            # Esegui i tool delle metriche direttamente
            results_for_user = {}
            tasks = []
            tool_names = [] # Tieni traccia dei nomi per l'associazione dei risultati
            
            for tool in current_metric_tools:
                if tool.coroutine:
                    metric_name = tool.name.replace("recommender_", "")
                    tool_names.append(metric_name) # Aggiungi il nome alla lista

                    # Seleziona il catalogo specifico per la metrica, altrimenti usa il fallback
                    current_catalog_json = metric_specific_catalogs.get(metric_name, catalog_json_fallback)
                    print(f"Invoking tool '{tool.name}' with {'specific' if metric_name in metric_specific_catalogs else 'fallback'} catalog.")

                    # Crea il task per la coroutine del tool
                    tasks.append(tool.coroutine(catalog=current_catalog_json, user_profile=profile_summary))

            if tasks:
                metric_outputs = await asyncio.gather(*tasks, return_exceptions=True)
                for i, output in enumerate(metric_outputs):
                    metric_name = tool_names[i] # Usa la lista di nomi per associare correttamente
                    if isinstance(output, Exception): results_for_user[metric_name] = {"metric": metric_name, "recommendations": [], "explanation": f"Error: {output}"}
                    elif isinstance(output, dict): results_for_user[metric_name] = output
                    else: results_for_user[metric_name] = {"metric": metric_name, "recommendations": [], "explanation": f"Unexpected type: {type(output)}"}
            user_metric_results[user_id] = results_for_user
            end_user = time.time()
            print(f"Tempo impiegato per utente {user_id}: {end_user - start_user:.2f} secondi")
        end_all_users = time.time()
        print(f"Tempo totale per generazione raccomandazioni di tutti gli utenti: {end_all_users - start_all_users:.2f} secondi")

        # Valutazione aggregata
        print("\n--- Valutazione Aggregata ---")
        final_evaluation = {}
        if user_metric_results:
            try:
                start_json = time.time()
                all_results_str = json.dumps(user_metric_results, ensure_ascii=False, indent=2)
                end_json = time.time()
                print(f"Tempo per serializzazione JSON dei risultati: {end_json - start_json:.2f} secondi")

                eval_catalog = self.get_optimized_catalog(limit=300)

                start_eval = time.time()
                if self.evaluator_tool and self.evaluator_tool.coroutine:
                    start_eval = time.time()
                    final_evaluation = await self.evaluator_tool.coroutine(all_recommendations_str=all_results_str, catalog_str=eval_catalog)
                    end_eval = time.time()
                    print(f"Tempo impiegato per valutazione aggregata: {end_eval - start_eval:.2f} secondi")
                else: raise ValueError("Evaluator tool/coroutine non definito.")
                end_eval = time.time()
                print(f"Tempo impiegato per valutazione aggregata: {end_eval - start_eval:.2f} secondi")
                
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
            # Converti np.float64 in float nativo per JSON
            def convert_np_float(obj):
                if isinstance(obj, np.float64): return float(obj)
                if isinstance(obj, dict): return {k: convert_np_float(v) for k, v in obj.items()}
                if isinstance(obj, list): return [convert_np_float(i) for i in obj]
                return obj
            result_data["metrics"] = convert_np_float(metrics_calculated)
            
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
        k_values = [1, 5, 10, 20, 50] # Valori di K per cui calcolare la precisione
        
        # Calcola precision@k per le raccomandazioni finali vs *tutti* gli item hold-out aggregati
        # (Manteniamo questa metrica aggregata per le final recs, dato che sono aggregate)
        all_relevant_items_flat = [item for sublist in per_user_relevant_items.values() for item in sublist]
        final_pak_scores = {k: calculate_precision_at_k(all_final_recs, all_relevant_items_flat, k=k) for k in k_values}
        
        # Metriche per utente
        metric_names = list(self.current_prompt_variants.keys()) # Ottiene nomi metriche (es. precision_at_k, coverage)
        # Aggiorna struttura per tenere score per ogni K
        aggregated_metrics = {name: {'precision_scores': {k: [] for k in k_values}, 'genre_coverage_scores': []} for name in metric_names}
        aggregated_metrics['final'] = {'precision_scores': final_pak_scores, 'genre_coverage_scores': []} # Precision finale è già calcolata

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
                
                # Calcola Precision@k per diversi k
                pak_scores = {k: calculate_precision_at_k(recs, user_relevant, k=k) for k in k_values}
                
                # Genre Coverage (calcolata sulla lista completa di recs)
                genres = set() 
                if self.movies is not None:
                    def get_genres(mid): 
                        m = self.movies[self.movies['movie_id'] == mid]
                        return set(m.iloc[0]['genres'].split('|')) if not m.empty and pd.notna(m.iloc[0]['genres']) else set()
                    # Usa tutti i recs generati per calcolare la coverage dei generi
                    genres = set().union(*[get_genres(mid) for mid in recs]) 
                    all_available_genres = set(g for movie_genres in self.movies['genres'].dropna() for g in movie_genres.split('|'))
                    n_genres = len(all_available_genres)
                    genre_cov = len(genres) / n_genres if n_genres > 0 else 0.0
                else:
                    genre_cov = 0.0

                user_metrics_calculated[metric_name] = {
                    "precision_scores": pak_scores, # Ora è un dizionario
                    "genre_coverage": genre_cov
                }
                # Aggiungi i punteggi P@k all'aggregazione
                for k, score in pak_scores.items():
                    aggregated_metrics[metric_name]['precision_scores'][k].append(score)
                
                aggregated_metrics[metric_name]['genre_coverage_scores'].append(genre_cov)
                # Stampa P@k per valori selezionati
                pak_str = ", ".join([f"P@{k}={score:.4f}" for k, score in pak_scores.items()])
                print(f"    {metric_name}: {pak_str}, GenreCoverage={genre_cov:.4f}")

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
            # Calcola Mean Average Precision per ogni k
            map_at_k_scores = {k: np.mean(aggregated_metrics[name]['precision_scores'][k]) 
                               if aggregated_metrics[name]['precision_scores'][k] else 0.0 
                               for k in k_values}
                               
            avg_gen_cov = np.mean(aggregated_metrics[name]['genre_coverage_scores']) if aggregated_metrics[name]['genre_coverage_scores'] else 0.0
            label = f"Mean {name.capitalize()}" if name != 'final' else "Final Aggregated"
            
            map_str = ", ".join([f"MAP@{k}={score:.4f}" for k, score in map_at_k_scores.items()])
            # Per 'final', usiamo i valori aggregati già calcolati
            if name == 'final':
                 final_pak_str = ", ".join([f"P@{k}={score:.4f}" for k, score in final_pak_scores.items()])
                 print(f"  {label}: {final_pak_str} (vs all held-out), GenreCoverage={final_genre_cov_agg:.4f}")
                 final_metrics_summary["aggregate_mean"][name] = {"precision_scores_agg": final_pak_scores, "genre_coverage": final_genre_cov_agg}
            else:
                 print(f"  {label}: {map_str}, Mean GenreCoverage={avg_gen_cov:.4f}")
                 final_metrics_summary["aggregate_mean"][name] = {"map_at_k": map_at_k_scores, "mean_genre_coverage": avg_gen_cov}
        
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
        # Converti np.float64 in float nativo per JSON prima di salvare
        def convert_np_float_exp(obj):
            if isinstance(obj, np.float64): return float(obj)
            if isinstance(obj, dict): return {k: convert_np_float_exp(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert_np_float_exp(i) for i in obj]
            return obj
        result_to_save = convert_np_float_exp(result)
        
        try:
            with open(filename, "w", encoding="utf-8") as f: json.dump(result_to_save, f, ensure_ascii=False, indent=2)
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
        
        # Converti np.float64 in float nativo prima di restituire
        def convert_np_float_ret(obj):
            if isinstance(obj, np.float64): return float(obj)
            if isinstance(obj, dict): return {k: convert_np_float_ret(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert_np_float_ret(i) for i in obj]
            return obj
            
        return convert_np_float_ret({"timestamp": datetime.now().isoformat(), "metric_recommendations": metric_results, "final_evaluation": final_evaluation, "metrics": metrics}) # Restituisce nuove metriche 