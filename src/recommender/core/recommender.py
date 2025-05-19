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
from pydantic import ValidationError # NUOVO: Aggiunto per la validazione Pydantic

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
    "max_tokens": 2500, # AUMENTATO da 2500 per permettere output più lunghi (50 recs + spiegazione)
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
        
        print(f"DEBUG _run_metric_tool_internal: Called for metric '{metric_name}', user {user_id}") # NUOVA PRINT
        metric_desc = self.current_prompt_variants.get(metric_name)
        print(f"DEBUG _run_metric_tool_internal: metric_desc = {repr(metric_desc)}") # NUOVA PRINT
        
        prompt_template = create_metric_prompt(metric_name, metric_desc)
        print(f"DEBUG _run_metric_tool_internal: prompt_template type = {type(prompt_template)}") # NUOVA PRINT
        if hasattr(prompt_template, 'template'):
            print(f"DEBUG _run_metric_tool_internal: prompt_template.template (first 100 chars) = {repr(prompt_template.template[:100])}") # NUOVA PRINT
        
        # NUOVA PRINT
        print(f"DEBUG _run_metric_tool_internal: repr(prompt_template) = {repr(prompt_template)}")
        print(f"DEBUG _run_metric_tool_internal: prompt_template.input_variables = {prompt_template.input_variables if hasattr(prompt_template, 'input_variables') else 'N/A'}")

        for attempt in range(max_attempts):
            try: # Blocco try principale
                prompt_str = None # Inizializza
                try: # NUOVO TRY-EXCEPT SPECIFICO
                    print(f"DEBUG _run_metric_tool_internal: Attempting prompt_template.format() for metric '{metric_name}', user {user_id}, attempt {attempt + 1}") # NUOVA PRINT
                    prompt_str = prompt_template.format(catalog=catalog, user_profile=user_profile)
                    print(f"DEBUG _run_metric_tool_internal: prompt_template.format() successful. prompt_str (first 100 chars) = {repr(prompt_str[:100])}") # NUOVA PRINT
                except Exception as format_exception:
                    print(f"!!! ERROR during prompt_template.format() for metric '{metric_name}', user {user_id}, attempt {attempt + 1} !!!")
                    print(f"  Exception type: {type(format_exception)}")
                    print(f"  Exception repr: {repr(format_exception)}")
                    print(f"  Exception str: {str(format_exception)}")
                    print(f"  catalog (type: {type(catalog)}, len: {len(catalog) if isinstance(catalog, (str, list, dict)) else 'N/A'}): {repr(catalog[:200]) if isinstance(catalog, str) else catalog}")
                    print(f"  user_profile (type: {type(user_profile)}, len: {len(user_profile) if isinstance(user_profile, (str, list, dict)) else 'N/A'}): {repr(user_profile[:200]) if isinstance(user_profile, str) else user_profile}")
                    if hasattr(prompt_template, 'template'):
                         print(f"  prompt_template.template (full): {repr(prompt_template.template)}")
                    raise format_exception # Rilancia per essere catturata dal blocco esterno

                print(f"Attempt {attempt+1}/{max_attempts} invoking LLM for metric: {metric_name} (user {user_id if user_id is not None else 'N/A'}) - expecting JSON in backticks.")
                
                # MODIFICATO: Chiamata LLM diretta per ottenere testo grezzo
                text_content = None # Inizializza text_content
                try:
                    raw_response_message = await self.llm.ainvoke(prompt_str, max_tokens=3000)
                    text_content = getattr(raw_response_message, 'content', str(raw_response_message))
                    print(f"DEBUG metric {metric_name} (user {user_id if user_id is not None else 'N/A'}, attempt {attempt+1}): LLM call successful. Raw response type: {type(raw_response_message)}, text_content type: {type(text_content)}")
                except Exception as llm_call_exception:
                    user_id_str = str(user_id) if user_id is not None else 'N/A'
                    print(f"!!! ERROR during LLM call for metric {metric_name} (user {user_id_str}, attempt {attempt+1}) !!!")
                    print(f"  Exception type: {type(llm_call_exception)}")
                    print(f"  Exception repr: {repr(llm_call_exception)}")
                    print(f"  Exception str: {str(llm_call_exception)}")
                    # Rilancia l'eccezione per essere gestita dal blocco try-except esterno esistente
                    # o gestiscila specificamente qui se necessario
                    raise llm_call_exception 

                # NUOVO: Estrazione JSON da triple backticks
                json_str = None
                # Prova prima ```json ... ```
                match = re.search(r"```json\s*(.*?)\s*```", text_content, re.DOTALL | re.IGNORECASE)
                if match:
                    json_str = match.group(1)
                else:
                    # Prova ``` ... ```
                    match = re.search(r"```\s*(.*?)\s*```", text_content, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                    else:
                        user_id_str = str(user_id) if user_id is not None else 'N/A'
                        # DEBUG PRINT AGGIUNTO - CORRETTO COME F-STRING MULTI-RIGA
                        print(f"""DEBUG metric {metric_name} (user {user_id_str}, attempt {attempt+1}): LLM raw output (text_content):
{text_content[:1000]}...
""") 
                        raise ValueError("No JSON block found enclosed in ```json ... ``` or ``` ... ``` backticks.")

                # NUOVO: Logica per rimuovere virgolette esterne se presenti
                unquoted_json_str = json_str
                if isinstance(json_str, str):
                    if (json_str.startswith("'") and json_str.endswith("'")) or \
                       (json_str.startswith('"') and json_str.endswith('"')):
                        unquoted_json_str = json_str[1:-1]
                        print(f"DEBUG metric {metric_name} (user {user_id if user_id is not None else 'N/A'}, attempt {attempt+1}): Removed surrounding quotes. Original: {repr(json_str)}, Unquoted: {repr(unquoted_json_str)}")

                print(f"DEBUG metric {metric_name} (user {user_id if user_id is not None else 'N/A'}, attempt {attempt+1}): json_str (potentially unquoted) BEFORE strip: {repr(unquoted_json_str)}")
                stripped_json_str = unquoted_json_str.strip()
                print(f"DEBUG metric {metric_name} (user {user_id if user_id is not None else 'N/A'}, attempt {attempt+1}): json_str AFTER strip: {repr(stripped_json_str)}")

                if not stripped_json_str: # Modificato per controllare stripped_json_str
                    raise ValueError("Extracted JSON string is empty after strip.")
                    
                try:
                    parsed_json = json.loads(stripped_json_str)
                except json.JSONDecodeError as e:
                    user_id_str = str(user_id) if user_id is not None else 'N/A'
                    # DEBUG PRINT AGGIUNTI
                    print(f"DEBUG metric {metric_name} (user {user_id_str}, attempt {attempt+1}): FAILED JSON PARSE.")
                    print(f"  text_content[:500]: {text_content[:500]}...")
                    print(f"  json_str REPR: {repr(json_str)}")
                    print(f"  stripped_json_str REPR: {repr(stripped_json_str)}")
                    print(f"  Exception type: {type(e)}")
                    print(f"  Exception repr: {repr(e)}")
                    print(f"  Exception msg: {e.msg}")
                    print(f"  Exception doc: {e.doc}") # Aggiunto per vedere il documento JSON che ha causato l'errore
                    print(f"  Exception pos: {e.pos}") # Aggiunto per vedere la posizione dell'errore
                    raise ValueError(f"Failed to decode JSON: {e.msg}. Initial char: '{stripped_json_str[:10] if stripped_json_str else ""}'. Extracted string repr: '{repr(stripped_json_str)[:200]}...'")

                try:
                    # Valida con Pydantic
                    # Assumendo che RecommendationOutput.parse_obj sia il metodo corretto per la versione Pydantic
                    recommendation_obj = RecommendationOutput.parse_obj(parsed_json)
                except ValidationError as e:
                    user_id_str = str(user_id) if user_id is not None else 'N/A'
                    print(f"DEBUG: LLM raw output (Pydantic validation failed) for metric {metric_name} (user {user_id_str}, attempt {attempt+1}): Content='{text_content[:500]}...' Parsed JSON='{str(parsed_json)[:500]}...'")
                    raise ValueError(f"Pydantic validation failed for RecommendationOutput: {e}. Parsed JSON: '{str(parsed_json)[:200]}...'")
                
                if len(recommendation_obj.recommendations) != NUM_RECOMMENDATIONS:
                    raise ValueError(f"Post-Pydantic validation failed: expected {NUM_RECOMMENDATIONS} items, found {len(recommendation_obj.recommendations)}.")
                
                return {"metric": metric_name, **recommendation_obj.dict()}
                
            except Exception as e:
                error_message_raw = str(e)
                # Esegui l'escape delle parentesi graffe per evitare problemi con f-string/template successivi
                error_message_escaped = error_message_raw.replace('{', '{{').replace('}', '}}')

                error_message_for_llm = error_message_escaped # Default a errore escapato
                parsed_json_str_for_feedback = str(parsed_json)[:100] if 'parsed_json' in locals() else 'N/A'

                if isinstance(e, ValueError):
                    if "No JSON block found" in error_message_raw:
                        error_message_for_llm = f"Your previous response did not contain a JSON object enclosed in triple backticks (e.g., ```json\\n{{{{...}}}}\\n```). Please provide the JSON in the correct format. Raw response started with: {text_content[:100].replace('{', '{{').replace('}', '}}')}"
                    elif "Failed to decode JSON" in error_message_raw:
                        error_message_for_llm = f"Your previous response contained a malformed JSON object within the triple backticks. Error: {error_message_escaped}. Please ensure the JSON is valid. The problematic JSON string started with: {json_str[:100].replace('{', '{{').replace('}', '}}')}"
                    elif "Pydantic validation failed" in error_message_raw or "Post-Pydantic validation failed" in error_message_raw:
                        specific_error_detail = error_message_escaped
                        if 'parsed_json' in locals() and isinstance(parsed_json.get('recommendations'), list):
                            actual_len = len(parsed_json['recommendations'])
                            specific_error_detail += f" Your list had {actual_len} items, but EXACTLY {NUM_RECOMMENDATIONS} are required."
                        error_message_for_llm = f"The JSON in your previous response was valid, but did not match the required schema (e.g., wrong field types, missing fields, or incorrect number of recommendations - expected exactly {NUM_RECOMMENDATIONS}). Error: {specific_error_detail}. Please correct the structure. Parsed JSON started with: {parsed_json_str_for_feedback.replace('{', '{{').replace('}', '}}')}"
                    elif "Extracted JSON string is empty" in error_message_raw:
                        error_message_for_llm = f"Your previous response resulted in an empty JSON string after extracting from backticks. Please provide a non-empty JSON object. Raw response started with: {text_content[:100].replace('{', '{{').replace('}', '}}')}"
                
                if attempt < max_attempts - 1:
                    print(f"Error in attempt {attempt+1} for metric {metric_name} (user {user_id if user_id is not None else 'N/A'}): {e}, retrying with feedback...")
                    
                    # Ricostruisci il prompt template con il feedback
                    current_template_str = prompt_template.template
                    # Assicurati che <|eot_id|> sia seguito da \n prima di <|start_header_id|>user
                    # Questo è un punto comune di inserimento per il feedback
                    # L'obiettivo è inserire un nuovo turno user/assistant con il feedback
                    # prima della richiesta finale all'assistant di generare l'output
                    
                    # Trova il punto di inserimento del feedback.
                    # Idealmente, dopo l'ultimo <|eot_id|> del system/user e prima del <|start_header_id|>assistant<|end_header_id|>
                    # che precede la risposta attesa.
                    insertion_point = "<|start_header_id|>assistant<|end_header_id|>\\n"
                    if insertion_point in current_template_str:
                         feedback_prompt_segment = (
                            f"<|eot_id|>\\n<|start_header_id|>user<|end_header_id|>\\n"
                            f"# IMPORTANT FEEDBACK ON PREVIOUS ATTEMPT:\\n"
                            f"{error_message_for_llm}\\n"
                            f"Please regenerate your response, ensuring it is a single JSON object enclosed in triple backticks, correctly structured, and contains exactly {NUM_RECOMMENDATIONS} recommendations.\\n"
                            f"<|eot_id|>\\n{insertion_point}"
                        )
                         updated_template_str = current_template_str.replace(
                            insertion_point, 
                            feedback_prompt_segment,
                            1 # Replace only the last occurrence before expected output
                         )
                    else: # Fallback se il punto di inserimento non è standard
                        updated_template_str = f"{current_template_str}\\n<|start_header_id|>user<|end_header_id|>\\n# FEEDBACK: {error_message_for_llm}<|eot_id|>\\n<|start_header_id|>assistant<|end_header_id|>\\n"

                    prompt_template = PromptTemplate(
                        input_variables=["catalog", "user_profile"], 
                        template=updated_template_str
                    )
                    continue 
                else:
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
        
        print(f"DEBUG _evaluate_recommendations_internal: Called.") # NUOVA PRINT
        full_eval_prompt_template = create_evaluation_prompt()
        print(f"DEBUG _evaluate_recommendations_internal: full_eval_prompt_template type = {type(full_eval_prompt_template)}") # NUOVA PRINT
        if hasattr(full_eval_prompt_template, 'template'):
            print(f"DEBUG _evaluate_recommendations_internal: full_eval_prompt_template.template (first 100 chars) = {repr(full_eval_prompt_template.template[:100])}") # NUOVA PRINT
            
        # NUOVA PRINT
        print(f"DEBUG _evaluate_recommendations_internal: repr(full_eval_prompt_template) = {repr(full_eval_prompt_template)}")
        print(f"DEBUG _evaluate_recommendations_internal: full_eval_prompt_template.input_variables = {full_eval_prompt_template.input_variables if hasattr(full_eval_prompt_template, 'input_variables') else 'N/A'}")

        feedback_for_llm_str = "" 

        for attempt in range(max_attempts):
            try: # Blocco try principale
                prompt_str = None # Inizializza
                try: # NUOVO TRY-EXCEPT SPECIFICO
                    print(f"DEBUG _evaluate_recommendations_internal: Attempting full_eval_prompt_template.format(), attempt {attempt + 1}") # NUOVA PRINT
                    prompt_str = full_eval_prompt_template.format(
                        all_recommendations=all_recommendations_str,
                        catalog=catalog_str,
                        feedback_block=feedback_for_llm_str 
                    )
                    print(f"DEBUG _evaluate_recommendations_internal: full_eval_prompt_template.format() successful. prompt_str (first 100 chars) = {repr(prompt_str[:100])}") # NUOVA PRINT
                except Exception as format_exception:
                    print(f"!!! ERROR during full_eval_prompt_template.format(), attempt {attempt + 1} !!!")
                    print(f"  Exception type: {type(format_exception)}")
                    print(f"  Exception repr: {repr(format_exception)}")
                    print(f"  Exception str: {str(format_exception)}")
                    print(f"  all_recommendations_str (type: {type(all_recommendations_str)}, len: {len(all_recommendations_str) if isinstance(all_recommendations_str, str) else 'N/A'}): {repr(all_recommendations_str[:200])}")
                    print(f"  catalog_str (type: {type(catalog_str)}, len: {len(catalog_str) if isinstance(catalog_str, str) else 'N/A'}): {repr(catalog_str[:200])}")
                    print(f"  feedback_for_llm_str (type: {type(feedback_for_llm_str)}, len: {len(feedback_for_llm_str) if isinstance(feedback_for_llm_str, str) else 'N/A'}): {repr(feedback_for_llm_str)}")
                    if hasattr(full_eval_prompt_template, 'template'):
                         print(f"  full_eval_prompt_template.template (full): {repr(full_eval_prompt_template.template)}")
                    raise format_exception # Rilancia per essere catturata dal blocco esterno

                print(f"Attempt {attempt+1}/{max_attempts} invoking evaluation aggregator - expecting JSON in backticks.")
                
                # MODIFICATO: Chiamata LLM diretta per ottenere testo grezzo
                text_content = None # Inizializza text_content
                try:
                    raw_response_message = await self.llm.ainvoke(prompt_str, max_tokens=3000)
                    text_content = getattr(raw_response_message, 'content', str(raw_response_message))
                    print(f"DEBUG evaluation (attempt {attempt+1}): LLM call successful. Raw response type: {type(raw_response_message)}, text_content type: {type(text_content)}")
                except Exception as llm_call_exception:
                    print(f"!!! ERROR during LLM call for evaluation (attempt {attempt+1}) !!!")
                    print(f"  Exception type: {type(llm_call_exception)}")
                    print(f"  Exception repr: {repr(llm_call_exception)}")
                    print(f"  Exception str: {str(llm_call_exception)}")
                    raise llm_call_exception

                # NUOVO: Estrazione JSON da triple backticks
                json_str = None
                match = re.search(r"```json\s*(.*?)\s*```", text_content, re.DOTALL | re.IGNORECASE)
                if match:
                    json_str = match.group(1)
                else:
                    # Prova ``` ... ```
                    match = re.search(r"```\s*(.*?)\s*```", text_content, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                    else:
                        # DEBUG PRINT AGGIUNTO - CORRETTO COME F-STRING MULTI-RIGA
                        print(f"""DEBUG evaluation (attempt {attempt+1}): LLM raw output (text_content):
{text_content[:1000]}...
""")
                        raise ValueError("No JSON block found in LLM response for evaluation, or backticks missing.")
                
                # NUOVO: Logica per rimuovere virgolette esterne se presenti
                unquoted_json_str = json_str
                if isinstance(json_str, str):
                    if (json_str.startswith("'") and json_str.endswith("'")) or \
                       (json_str.startswith('"') and json_str.endswith('"')):
                        unquoted_json_str = json_str[1:-1]
                        print(f"DEBUG evaluation (attempt {attempt+1}): Removed surrounding quotes. Original: {repr(json_str)}, Unquoted: {repr(unquoted_json_str)}")

                print(f"DEBUG evaluation (attempt {attempt+1}): json_str (potentially unquoted) BEFORE strip: {repr(unquoted_json_str)}")
                stripped_json_str = unquoted_json_str.strip()
                print(f"DEBUG evaluation (attempt {attempt+1}): json_str AFTER strip: {repr(stripped_json_str)}")

                if not stripped_json_str: # Modificato per controllare stripped_json_str
                    raise ValueError("Extracted JSON string for evaluation is empty after strip.")

                try:
                    parsed_json = json.loads(stripped_json_str)
                except json.JSONDecodeError as e:
                    # DEBUG PRINT AGGIUNTI
                    print(f"DEBUG evaluation (attempt {attempt+1}): FAILED JSON PARSE.")
                    print(f"  text_content[:500]: {text_content[:500]}...")
                    print(f"  json_str REPR: {repr(json_str)}")
                    print(f"  stripped_json_str REPR: {repr(stripped_json_str)}")
                    print(f"  Exception type: {type(e)}")
                    print(f"  Exception repr: {repr(e)}")
                    print(f"  Exception msg: {e.msg}")
                    print(f"  Exception doc: {e.doc}") # Aggiunto
                    print(f"  Exception pos: {e.pos}") # Aggiunto
                    raise ValueError(f"Failed to decode JSON for evaluation: {e.msg}. Initial char: '{stripped_json_str[:10] if stripped_json_str else ""}'. Extracted string repr: '{repr(stripped_json_str)[:200]}...'")

                try:
                    # Valida con Pydantic
                    evaluation_obj = EvaluationOutput.parse_obj(parsed_json)
                except ValidationError as e:
                    print(f"DEBUG: LLM raw output (Pydantic validation failed) for evaluation (attempt {attempt+1}): Content='{text_content[:500]}...' Parsed JSON='{str(parsed_json)[:500]}...'")
                    raise ValueError(f"Pydantic validation failed for EvaluationOutput: {e}. Parsed JSON: '{str(parsed_json)[:200]}...'")

                if len(evaluation_obj.final_recommendations) != NUM_RECOMMENDATIONS:
                    raise ValueError(f"Post-Pydantic validation failed for evaluation: expected {NUM_RECOMMENDATIONS} items, found {len(evaluation_obj.final_recommendations)}")
                
                return evaluation_obj.dict()
            
            except Exception as e:
                error_message_raw = str(e)
                # Esegui l'escape delle parentesi graffe per evitare problemi con f-string/template successivi
                error_message_escaped = error_message_raw.replace('{', '{{').replace('}', '}}')
                
                error_message_for_llm = error_message_escaped # Default a errore escapato
                parsed_json_str_for_feedback = str(parsed_json)[:100] if 'parsed_json' in locals() else 'N/A'

                if isinstance(e, ValueError):
                    if "No JSON block found" in error_message_raw:
                        error_message_for_llm = f"Your previous evaluation response did not contain a JSON object enclosed in triple backticks (e.g., ```json\\n{{{{...}}}}\\n```). Please provide the JSON in the correct format. Raw response started with: {text_content[:100].replace('{', '{{').replace('}', '}}')}"
                    elif "Failed to decode JSON" in error_message_raw:
                        error_message_for_llm = f"Your previous evaluation response contained a malformed JSON object. Error: {error_message_escaped}. Please ensure the JSON is valid. The problematic JSON string started with: {json_str[:100].replace('{', '{{').replace('}', '}}')}"
                    elif "Pydantic validation failed" in error_message_raw or "Post-Pydantic validation failed" in error_message_raw:
                        specific_error_detail = error_message_escaped
                        if 'parsed_json' in locals() and isinstance(parsed_json.get('final_recommendations'), list):
                            actual_len = len(parsed_json['final_recommendations'])
                            specific_error_detail += f" Your list had {actual_len} items, but EXACTLY {NUM_RECOMMENDATIONS} are required."
                        error_message_for_llm = f"The JSON in your previous evaluation response was valid, but did not match the required schema (e.g., missing fields, or incorrect number of final_recommendations - expected {NUM_RECOMMENDATIONS}). Error: {specific_error_detail}. Please correct the structure. Parsed JSON started with: {parsed_json_str_for_feedback.replace('{', '{{').replace('}', '}}')}"
                    elif "Extracted JSON string for evaluation is empty" in error_message_raw:
                        error_message_for_llm = f"Your previous response resulted in an empty JSON string after extracting from backticks. Please provide a non-empty JSON object. Raw response started with: {text_content[:100].replace('{', '{{').replace('}', '}}')}"


                if attempt < max_attempts - 1:
                    print(f"Error in evaluation attempt {attempt+1}: {e}, retrying with feedback...")
                    # Il feedback viene inserito nel placeholder {feedback_block} del prompt di valutazione
                    feedback_for_llm_str = (
                        f"\\n\\n# IMPORTANT FEEDBACK ON PREVIOUS ATTEMPT (Attempt {attempt+1}):\\n"
                        f"{error_message_for_llm}\\n"
                        f"Please regenerate your response, ensuring it is a single JSON object enclosed in triple backticks, correctly structured, and contains exactly {NUM_RECOMMENDATIONS} final recommendations."
                    )
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