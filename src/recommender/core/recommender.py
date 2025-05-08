"""
Sistema di raccomandazione multi-metrica basato su LangGraph.

Note sulla migrazione da AgentExecutor a LangGraph:
-------------------------------------------------
Questa versione del sistema utilizza LangGraph al posto del vecchio AgentExecutor di LangChain.
I principali cambiamenti sono:

1. Definizione esplicita del flusso di raccomandazione come un grafo di stati
2. Orchestrazione parallela delle metriche con convergenza controllata
3. Gestione più robusta dello stato e delle transizioni
4. Controllo più granulare sul processo decisionale

La logica di base rimane invariata:
- Ogni utente viene elaborato in sequenza
- Per ogni utente, le metriche precision_at_k e coverage vengono calcolate in parallelo
- I risultati per tutti gli utenti sono poi aggregati in una valutazione finale

I metodi per l'interfaccia pubblica (generate_standard_recommendations, 
generate_recommendations_with_custom_prompt, ecc.) rimangono invariati. 
"""

import os
import json
import pandas as pd
import numpy as np # Aggiunto per la media delle metriche
import re
import sys  # Aggiunto per sys.stdout.flush()
import traceback # Aggiunto per debug
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import time

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# LangGraph imports (NUOVO)
from langgraph.graph import StateGraph, END

# Moduli locali
from src.recommender.utils.data_processor import (
    load_ratings, 
    load_movies, 
    filter_users_by_specific_users,
    create_user_profiles
)
from src.recommender.utils.rag_utils import MovieRAG
from src.recommender.utils.data_utils import convert_numpy_types_for_json

# Modifica l'importazione per rimuovere calculate_precision_at_k e calculate_coverage
from src.recommender.core.metrics_utils import MetricsCalculator

# Importazione dal nuovo modulo prompt_manager
from src.recommender.core.prompt_manager import (
    PROMPT_VARIANTS, 
    NUM_RECOMMENDATIONS, 
    create_metric_prompt, 
    create_evaluation_prompt
)

# Importazione della nuova classe di nodi per LangGraph
from src.recommender.core.graph_nodes import RecommenderGraphNodes, RecommenderState

# Importazione degli schemi Pydantic spostati in un modulo dedicato
from src.recommender.core.schemas import RecommendationOutput, EvaluationOutput

# Importazione del gestore degli esperimenti
from src.recommender.core.experiment_manager import ExperimentManager

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
    "temperature": 0.2,
    "max_tokens": 2500, # AUMENTATO da 1536 per permettere output più lunghi (50 recs + spiegazione)
}


LLM_MODEL_ID = "google/gemini-2.5-flash-preview"

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


class RecommenderSystem:
    """
    Sistema di raccomandazione unificato basato su LangGraph.
    """
    
    def __init__(self, specific_user_ids: List[int] = [4277, 4169, 1680], model_id: str = LLM_MODEL_ID):
        self.specific_user_ids = specific_user_ids
        self.model_id = model_id
        self.llm = llm
        self.filtered_ratings = None
        self.user_profiles = None 
        self.movies = None
        self.rag = None
        self.recommender_graph = None  # MODIFICATO: sostituisce self.agent
        self.metric_tools = []         # Mantenuto per compatibilità
        self.evaluator_tool = None     # Mantenuto per compatibilità
        self.datasets_loaded = False
        self.current_prompt_variants = PROMPT_VARIANTS.copy() # Inizializza con default
        self.graph_nodes = None        # Riferimento alla classe dei nodi LangGraph
        self.experiment_manager = None # Gestore degli esperimenti

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

    def _initialize_langgraph(self) -> None:
        """Inizializza il grafo LangGraph per orchestrare il processo di raccomandazione."""
        print("Inizializzazione LangGraph...")
        
        # Inizializza la classe dei nodi del grafo
        self.graph_nodes = RecommenderGraphNodes(self)
        
        # Crea il grafo di stati
        workflow = StateGraph(RecommenderState)
        
        # Aggiungi nodi per ogni fase del processo
        # 1. Nodo di inizializzazione
        workflow.add_node("initialize", self.graph_nodes.node_initialize)
        
        # 2. Nodo per preparare dati utente
        workflow.add_node("prepare_user_data", self.graph_nodes.node_prepare_user_data)
        
        # 3. Nodi per le metriche 
        workflow.add_node("run_precision_metric", self.graph_nodes.node_run_precision_metric)
        workflow.add_node("run_coverage_metric", self.graph_nodes.node_run_coverage_metric)
        
        # 4. Nodo per raccogliere risultati metriche
        workflow.add_node("collect_metric_results", self.graph_nodes.node_collect_metric_results)
        
        # 5. Nodo per passare al prossimo utente o terminare
        workflow.add_node("next_user_or_finish", self.graph_nodes.node_next_user_or_finish)
        
        # 6. Nodo valutatore finale
        workflow.add_node("evaluate_all_results", self.graph_nodes.node_evaluate_all_results)
        
        # Definire il punto di ingresso
        workflow.set_entry_point("initialize")
        
        # Definire i flussi
        workflow.add_edge("initialize", "prepare_user_data")
        workflow.add_edge("prepare_user_data", "run_precision_metric")
        workflow.add_edge("prepare_user_data", "run_coverage_metric")
        
        # Rimuovo i bordi condizionali e aggiungo bordi diretti a collect_metric_results
        workflow.add_edge("run_precision_metric", "collect_metric_results")
        workflow.add_edge("run_coverage_metric", "collect_metric_results")
        
        # Dopo aver raccolto i risultati, decidere se passare all'utente successivo o concludere
        workflow.add_conditional_edges(
            "collect_metric_results",
            self.graph_nodes.check_users_completion,
            {
                "next_user": "prepare_user_data",  # Vai al prossimo utente
                "evaluate": "evaluate_all_results"  # Tutti gli utenti elaborati, valuta
            }
        )
        
        # L'evaluator conclude il grafo
        workflow.add_edge("evaluate_all_results", END)
        
        # Compila il grafo
        self.recommender_graph = workflow.compile()
        
        print("LangGraph inizializzato.")
    
    def _check_metrics_completion(self, state: RecommenderState) -> str:
        """Verifica se tutte le metriche sono state calcolate."""
        return self.graph_nodes.check_metrics_completion(state)

    def _check_users_completion(self, state: RecommenderState) -> str:
        """Verifica se tutti gli utenti sono stati elaborati."""
        return self.graph_nodes.check_users_completion(state)

    # Funzioni di supporto interne che riutilizzano la logica esistente
    async def _run_metric_tool_internal(self, metric_name: str, catalog: str, user_profile: str, user_id: Optional[int] = None) -> Dict:
        """Versione interna di run_metric_agent_tool che può essere chiamata dai nodi."""
        max_attempts = 3
        metric_desc = self.current_prompt_variants.get(metric_name)
        prompt_template = create_metric_prompt(metric_name, metric_desc)
        
        for attempt in range(max_attempts):
            try:
                prompt_str = prompt_template.format(catalog=catalog, user_profile=user_profile)
                
                # Seleziona il metodo per l'output strutturato
                structured_output_method = "function_calling"
                
                # Aggiungi schema più rigido e validatori 
                structured_llm = self.llm.with_structured_output(
                    RecommendationOutput, 
                    method=structured_output_method,
                    include_raw=True  # Per scopi di debug
                )
                
                print(f"Attempt {attempt+1}/{max_attempts} invoking structured LLM for metric: {metric_name}")
                
                # Aumenta il context window per evitare troncamenti
                parsed_response = await structured_llm.ainvoke(
                    prompt_str,
                    max_tokens=3000
                )
                
                # Accedi a 'parsed' dal dizionario restituito da ainvoke
                recommendation_obj = parsed_response.get('parsed') # Usa .get() per sicurezza

                if recommendation_obj is None:
                    parsing_error_details = parsed_response.get('parsing_error', 'N/A')
                    raw_output_aimessage = parsed_response.get('raw')
                    raw_content = getattr(raw_output_aimessage, 'content', '') if raw_output_aimessage else ''
                    
                    user_id_str = str(user_id) if user_id is not None else 'N/A'
                    print(f"DEBUG: LLM raw output for metric {metric_name} (user {user_id_str}, attempt {attempt+1}): AIMessage content='{str(raw_content)[:200]}...', additional_kwargs={getattr(raw_output_aimessage, 'additional_kwargs', {})}")

                    error_details_for_exception = str(parsing_error_details) if parsing_error_details and parsing_error_details != 'N/A' else "No specific parsing error details provided by the parser."
                    
                    # Controlla se l'output era vuoto o un rifiuto
                    is_empty_output_or_refusal = not raw_content and (not parsing_error_details or parsing_error_details == 'N/A' or "no specific parsing error" in error_details_for_exception.lower())
                    
                    if is_empty_output_or_refusal:
                        error_details_for_exception = "The LLM response was empty or a refusal. A valid structured output is MANDATORY."
                    
                    raise ValueError(f"LLM output could not be structured. Details: {error_details_for_exception}")

                # A questo punto, recommendation_obj non è None.
                # La validazione Pydantic (len == NUM_RECOMMENDATIONS e tipi int) è già avvenuta DENTRO with_structured_output.
                # La riga seguente è una doppia verifica, ora sicura.
                if len(recommendation_obj.recommendations) != NUM_RECOMMENDATIONS:
                    # Non dovremmo mai arrivare qui se il validatore Pydantic di RecommendationOutput ha funzionato.
                    raise ValueError(f"Post-structuring validation failed: expected {NUM_RECOMMENDATIONS} items, found {len(recommendation_obj.recommendations)} in a supposedly valid object.")
                
                return {"metric": metric_name, **recommendation_obj.dict()}
                
            except Exception as e:
                # Logica di gestione errori uguale a run_metric_agent_tool
                if attempt < max_attempts - 1:
                    print(f"Error in attempt {attempt+1} for metric {metric_name} (user {user_id if user_id is not None else 'N/A'}): {e}, retrying...")
                    
                    # Tentativo di rendere il retry più intelligente fornendo feedback all'LLM
                    error_feedback = ""
                    try:
                        # Se l'eccezione è la nostra ValueError da parsing fallito
                        if isinstance(e, ValueError) and "LLM output could not be structured" in str(e):
                            # Estrai il parsing_error specifico se disponibile
                            # Il messaggio è tipo: "LLM output could not be structured into RecommendationOutput. Parsing error: XYZ"
                            parts = str(e).split("Details: ")
                            if len(parts) > 1:
                                specific_error_msg = parts[1]
                                if specific_error_msg: 
                                    error_feedback = f"Your previous response for {metric_name} (attempt {attempt}) FAILED with the error: '{specific_error_msg}'. You MUST correct this. Ensure your output strictly follows all formatting rules, provides exactly {NUM_RECOMMENDATIONS} movie IDs, a valid explanation, and is a complete, non-empty response."
                    except Exception as extraction_error:
                        print(f"Could not extract specific error for feedback: {extraction_error}")

                    if not error_feedback: # Fallback a un messaggio di errore generico se non si estrae nulla
                        error_feedback = f"Your previous response for {metric_name} (attempt {attempt}) was not valid. Please try again, strictly following all output format rules (especially exactly {NUM_RECOMMENDATIONS} movie IDs, a valid explanation, and a complete, non-empty response)."

                    # Aggiungi il feedback al prompt template per il prossimo tentativo
                    # Assicurati che il template originale sia preservato e che il feedback sia aggiunto in modo appropriato
                    # Questo potrebbe richiedere di riformattare il prompt template originale se diventa troppo lungo
                    # o di inserire il feedback in un punto strategico.
                    # Per ora, lo aggiungiamo alla fine del template esistente, sperando che l'LLM lo consideri.
                    
                    # Ricostruisci il prompt template con il feedback
                    # NOTA: Questo modifica il prompt_template per i tentativi successivi di QUESTO CICLO
                    # Bisogna fare attenzione a non renderlo permanentemente modificato per altre chiamate.
                    # Poiché create_metric_prompt viene chiamato ad ogni _run_metric_tool_internal, 
                    # il prompt_template viene rigenerato fresco, quindi questa modifica è locale al ciclo for.

                    current_template_str = prompt_template.template
                    updated_template_str = current_template_str.replace(
                        "<|start_header_id|>assistant<|end_header_id|>\\n", # Punto di inserimento prima della risposta dell'assistente
                        f"<|start_header_id|>user<|end_header_id|>\\n# FEEDBACK_ON_PREVIOUS_ATTEMPT: {error_feedback} Please regenerate your response correctly.<|eot_id|>\\n<|start_header_id|>assistant<|end_header_id|>\\n"
                    )
                    prompt_template = PromptTemplate(
                        input_variables=["catalog", "user_profile"], # Le variabili non cambiano
                        template=updated_template_str
                    )
                    continue # Passa al prossimo tentativo
                else:
                    # Fallback con placeholder se tutti i tentativi falliscono
                    print(f"All attempts failed for {metric_name} (user {user_id if user_id is not None else 'N/A'}): {e}")
                    placeholder_recs = [0] * NUM_RECOMMENDATIONS 
                    return {
                        "metric": metric_name, 
                        "recommendations": placeholder_recs, 
                        "explanation": f"Error after {max_attempts} attempts for user {user_id if user_id is not None else 'N/A'}: {str(e)}. Returning placeholder recommendations."
                    }

    async def _evaluate_recommendations_internal(self, all_recommendations_str: str, catalog_str: str) -> Dict:
        """Versione interna di evaluate_recommendations_tool che può essere chiamata dai nodi."""
        max_attempts = 3
        
        # Usa la funzione create_evaluation_prompt importata dal modulo prompt_manager
        full_eval_prompt_template = create_evaluation_prompt()
        
        feedback_str = "" # Inizializza stringa di feedback

        for attempt in range(max_attempts):
            try:
                # Crea il prompt con le raccomandazioni per ciascuna metrica e il feedback
                prompt_str = full_eval_prompt_template.format(
                    all_recommendations=all_recommendations_str,
                    catalog=catalog_str,
                    feedback_block=feedback_str # Passa il feedback corrente
                )
                
                structured_llm = self.llm.with_structured_output(
                    EvaluationOutput, 
                    method="function_calling",
                    include_raw=True
                )
                
                print(f"Attempt {attempt+1}/{max_attempts} invoking evaluation aggregator")
                
                parsed_response = await structured_llm.ainvoke(prompt_str, max_tokens=3000)
                evaluation_obj = parsed_response.get('parsed')

                if evaluation_obj is None:
                    parsing_error_details = parsed_response.get('parsing_error', 'N/A')
                    raw_output_aimessage = parsed_response.get('raw')
                    raw_content = getattr(raw_output_aimessage, 'content', '') if raw_output_aimessage else ''

                    print(f"DEBUG: LLM raw output for evaluation (attempt {attempt+1}): AIMessage content='{str(raw_content)[:200]}...', additional_kwargs={getattr(raw_output_aimessage, 'additional_kwargs', {})}")
                    
                    error_details_for_exception = str(parsing_error_details) if parsing_error_details and parsing_error_details != 'N/A' else "No specific parsing error details provided by the parser."

                    is_empty_output_or_refusal = not raw_content and (not parsing_error_details or parsing_error_details == 'N/A' or "no specific parsing error" in error_details_for_exception.lower())

                    if is_empty_output_or_refusal:
                        error_details_for_exception = "The LLM response was empty or a refusal. A valid structured JSON output is MANDATORY."
                    
                    raise ValueError(f"LLM output for evaluation could not be structured. Details: {error_details_for_exception}")

                if len(evaluation_obj.final_recommendations) != NUM_RECOMMENDATIONS:
                    raise ValueError(f"Post-structuring validation failed for evaluation: expected {NUM_RECOMMENDATIONS} items, found {len(evaluation_obj.final_recommendations)}")
                
                return evaluation_obj.dict()
            
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"Error in evaluation attempt {attempt+1}: {e}, retrying...")
                    specific_error_msg = str(e) 
                    if isinstance(e, ValueError) and "LLM output for evaluation could not be structured" in str(e):
                         parts = str(e).split("Details: ")
                         if len(parts) > 1:
                            specific_error_msg = parts[1] # Usa il messaggio più dettagliato

                    feedback_str = f"\\n\\n# FEEDBACK_ON_PREVIOUS_ATTEMPT: Your previous response (attempt {attempt}) FAILED with the error: '{specific_error_msg}'. You MUST correct this. Ensure your output is a valid JSON object, with exactly {NUM_RECOMMENDATIONS} final recommendations, and is a complete, non-empty response."
                    continue
                else:
                    print(f"All evaluation attempts failed: {e}")
                    placeholder_recs = [0] * NUM_RECOMMENDATIONS
                    return {
                        "final_recommendations": placeholder_recs,
                        "justification": f"Error after {max_attempts} evaluation attempts: {str(e)}. Returning placeholder recommendations.",
                        "trade_offs": "Non disponibili a causa di errori."
                    }

    def initialize_system(self, force_reload_data: bool = False, force_recreate_vector_store: bool = False) -> None:
        """Metodo pubblico per inizializzare o reinizializzare il sistema."""
        print("\n=== Inizializzazione Sistema ===")
        self._load_datasets(force_reload=force_reload_data)
        self._initialize_rag(force_recreate_vector_store=force_recreate_vector_store)
        
        # MODIFICATO: usa _initialize_langgraph invece di _initialize_agent
        self._initialize_langgraph()
        
        # Inizializza il gestore degli esperimenti
        self.experiment_manager = ExperimentManager(self)
        
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
        Esegue l'intera pipeline di raccomandazione per tutti gli utenti specificati
        utilizzando LangGraph per l'orchestrazione.
        """
        # MODIFICATO: verifica che LangGraph sia inizializzato anziché l'agente
        if not self.recommender_graph:
            raise RuntimeError("LangGraph non inizializzato. Chiamare initialize_system() prima.")
        
        if not self.datasets_loaded or self.user_profiles is None or self.user_profiles.empty:
            raise RuntimeError("Dataset non caricati o profili utente vuoti. Chiamare initialize_system() prima.")

        # Imposta le varianti di prompt da usare per questa run
        self.current_prompt_variants = use_prompt_variants if use_prompt_variants is not None else PROMPT_VARIANTS.copy()
        
        start_all_users = time.time()
        
        # Stato iniziale per il grafo LangGraph
        initial_state = {
            "user_id": None,
            "user_profile": None,
            "catalog_precision": None,
            "catalog_coverage": None,
            "metric_results": {},
            "metric_tasks_completed": 0,
            "expected_metrics": len(self.current_prompt_variants),
            "all_user_results": {},
            "current_user_index": 0,
            "user_ids": self.specific_user_ids,
            "final_evaluation": None,
            "held_out_items": {},
            "error": None
        }
        
        try:
            # Esegui il grafo LangGraph
            print("\n=== Avvio pipeline con LangGraph ===")
            final_state = await self.recommender_graph.ainvoke(initial_state)
            
            # Estrai i risultati dal final_state
            user_metric_results = final_state["all_user_results"]
            final_evaluation = final_state["final_evaluation"]
            per_user_held_out_items = final_state["held_out_items"]
            
        except Exception as e:
            print(f"Errore nell'esecuzione del grafo LangGraph: {e}")
            traceback.print_exc()
            user_metric_results = {}
            final_evaluation = {"final_recommendations": [], "justification": f"Error: {e}", "trade_offs": "N/A"}
            per_user_held_out_items = {}
        
        end_all_users = time.time()
        print(f"Tempo totale per generazione raccomandazioni: {end_all_users - start_all_users:.2f} secondi")
        
        # Ripristina le varianti di prompt di default
        self.current_prompt_variants = PROMPT_VARIANTS.copy()
        
        return user_metric_results, final_evaluation, per_user_held_out_items

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
            result_data["metrics"] = convert_numpy_types_for_json(metrics_calculated)
            
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
        if not self.datasets_loaded or self.movies is None or self.movies.empty:
             print("Dataset non caricati o movies vuoti, metriche non calcolabili.")
             return {}
        
        # Inizializza MetricsCalculator
        metrics_calculator = MetricsCalculator(self.movies)
        
        k_values = [1, 5, 10, 20, 50]
        # Assicurati che current_prompt_variants sia popolato correttamente.
        # Fallback a PROMPT_VARIANTS di default se necessario.
        metric_names = list(self.current_prompt_variants.keys()) if self.current_prompt_variants else \
                       list(PROMPT_VARIANTS.keys())

        per_user_calculated_metrics, aggregate_calculated_metrics = metrics_calculator.compute_all_metrics(
            metric_results=metric_results,
            final_evaluation=final_evaluation,
            per_user_relevant_items=per_user_relevant_items,
            k_values=k_values,
            metric_names=metric_names
        )

        # --- Logica di Visualizzazione (utilizza i dati calcolati) ---
        print("\nMetriche Calcolate (Per Utente):")
        for user_id, u_calculated_data in per_user_calculated_metrics.items():
            # Stampa warning se l'utente non ha item rilevanti (opzionale, già gestito in parte)
            if not per_user_relevant_items.get(user_id, []):
                # Questo warning può essere utile per contesto, anche se P@k sarà 0
                print(f"  Utente {user_id}: Attenzione - Nessun item rilevante (held-out) di riferimento.")
            
            print(f"  Utente {user_id}:")
            for metric_name_key in metric_names: # Itera nell'ordine definito da metric_names
                data_for_metric = u_calculated_data.get(metric_name_key)
                if data_for_metric:
                    pak_scores = data_for_metric["precision_scores"]
                    genre_cov = data_for_metric["genre_coverage"]
                    pak_str = ", ".join([f"P@{k}={score:.4f}" for k, score in pak_scores.items()])
                    print(f"    {metric_name_key}: {pak_str}, GenreCoverage={genre_cov:.4f}")
                else:
                    # Se una metrica specifica non ha dati (dovrebbe essere raro se metric_names è corretto)
                    print(f"    {metric_name_key}: Dati non disponibili.")

        print("\nMetriche Aggregate (Medie su Utenti):")
        # Itera su metric_names + ['final'] per mantenere l'ordine e includere le metriche finali
        for name_key in metric_names + ['final']:
            agg_data = aggregate_calculated_metrics.get(name_key)
            if agg_data:
                label = f"Mean {name_key.capitalize()}" if name_key != 'final' else "Final Aggregated"
                
                if name_key == 'final':
                    pak_scores_agg = agg_data.get("precision_scores_agg", {})
                    genre_cov_agg = agg_data.get("genre_coverage", 0.0)
                    pak_str_agg = ", ".join([f"P@{k}={score:.4f}" for k, score in pak_scores_agg.items()])
                    print(f"  {label}: {pak_str_agg} (vs all held-out), GenreCoverage={genre_cov_agg:.4f}")
                else:
                    map_scores = agg_data.get("map_at_k", {})
                    mean_genre_cov = agg_data.get("mean_genre_coverage", 0.0)
                    map_str = ", ".join([f"MAP@{k}={score:.4f}" for k, score in map_scores.items()])
                    print(f"  {label}: {map_str}, Mean GenreCoverage={mean_genre_cov:.4f}")
            else:
                 print(f"  Metrica aggregata '{name_key}' non trovata.")

        total_item_cov = aggregate_calculated_metrics.get("total_item_coverage", 0.0)
        print(f"  Total Item Coverage (all recs): {total_item_cov:.4f}")

        final_metrics_summary = {
            "per_user": per_user_calculated_metrics,
            "aggregate_mean": aggregate_calculated_metrics
        }
        
        return final_metrics_summary
        
    # ----- Metodi per esperimenti ----- 
    async def generate_recommendations_with_custom_prompt(self, prompt_variants: Dict, experiment_name: str ="custom_experiment") -> Tuple[Dict, str]:
        """Genera raccomandazioni con un prompt personalizzato."""
        if not self.experiment_manager:
            self.experiment_manager = ExperimentManager(self)
        return await self.experiment_manager.run_experiment(prompt_variants, experiment_name)
        
    async def generate_standard_recommendations(self) -> Dict:
        """Genera raccomandazioni standard, calcola metriche e salva."""
        if not self.experiment_manager:
            self.experiment_manager = ExperimentManager(self)
        return await self.experiment_manager.run_standard_pipeline() 