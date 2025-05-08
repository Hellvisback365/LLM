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
import asyncio
import pandas as pd
import numpy as np # Aggiunto per la media delle metriche
import re
import sys  # Aggiunto per sys.stdout.flush()
import traceback # Aggiunto per debug
from datetime import datetime
from typing import Dict, List, Any, Tuple, TypedDict, Optional, cast
from dotenv import load_dotenv
import time

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from openai import RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# LangGraph imports (NUOVO)
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel

# Pydantic import
from pydantic import BaseModel, Field, field_validator

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

# ----------------------------
# Definizioni Prompt e Parser
# ----------------------------
NUM_RECOMMENDATIONS = 50 # Costante per il numero di raccomandazioni spostata prima delle classi Pydantic

# ----------------------------
# Definizione Stato LangGraph 
# ----------------------------
class RecommenderState(TypedDict):
    """Stato del grafo per il sistema di raccomandazione."""
    # Informazioni sull'utente corrente
    user_id: Optional[int]
    user_profile: Optional[str]
    
    # Cataloghi specifici per metrica
    catalog_precision: Optional[str]
    catalog_coverage: Optional[str]
    
    # Risultati delle metriche (separate keys per evitare concurrent updates)
    precision_at_k_result: Optional[Dict[str, Any]]
    coverage_result: Optional[Dict[str, Any]]
    precision_completed: bool  # Flag per tracking completion
    coverage_completed: bool   # Flag per tracking completion
    metric_results: Dict[str, Dict[str, Any]]
    metric_tasks_completed: int
    expected_metrics: int
    
    # Gestione multi-utente
    current_user_index: int
    user_ids: List[int]
    all_user_results: Dict[int, Dict[str, Dict[str, Any]]]
    
    # Output finale
    final_evaluation: Optional[Dict[str, Any]]
    
    # Dati per valutazione
    held_out_items: Dict[int, List[int]]
    
    # Gestione errori
    error: Optional[str]

# ----------------------------
# Definizioni Schemi Pydantic
# ----------------------------
class RecommendationOutput(BaseModel):
    """Schema per l'output dei tool di raccomandazione per metrica."""
    recommendations: List[int] = Field(
        ..., 
        description=f"Lista ORDINATA di esattamente {NUM_RECOMMENDATIONS} ID numerici di film raccomandati. Il primo ID è il più raccomandato, l'ultimo il meno.",
        min_items=NUM_RECOMMENDATIONS,
        max_items=NUM_RECOMMENDATIONS
    )
    explanation: str = Field(..., description="Breve spiegazione testuale del motivo per cui questi film sono stati scelti e ordinati in base alla metrica richiesta.")
    
    @field_validator('recommendations')
    def validate_exactly_50_items(cls, v):
        """Validatore che garantisce esattamente NUM_RECOMMENDATIONS elementi."""
        if len(v) != NUM_RECOMMENDATIONS:
            raise ValueError(f"L'array deve contenere esattamente {NUM_RECOMMENDATIONS} elementi, trovati {len(v)}")
        return v
    
    @field_validator('recommendations')
    def validate_ids_are_integers(cls, v_list: List[int]):
        """Validatore che garantisce che ogni elemento della lista sia un ID numerico intero."""
        for item in v_list:
            if not isinstance(item, int):
                raise ValueError(f"Ogni ID film deve essere un intero, trovato {type(item)}")
        return v_list

class EvaluationOutput(BaseModel):
    """Schema per l'output del tool di valutazione finale."""
    final_recommendations: List[int] = Field(
        ..., 
        description=f"Lista finale OTTIMALE e ORDINATA di esattamente {NUM_RECOMMENDATIONS} ID numerici di film, bilanciando le metriche. Il primo ID è il più raccomandato.",
        min_items=NUM_RECOMMENDATIONS,
        max_items=NUM_RECOMMENDATIONS
    )
    justification: str = Field(..., description="Spiegazione dettagliata della logica di selezione, bilanciamento e ORDINAMENTO per la lista finale aggregata.")
    trade_offs: str = Field(..., description="Descrizione dei trade-off considerati tra le diverse metriche (es. precisione vs copertura) nell'ordinamento finale.")
    
    @field_validator('final_recommendations')
    def validate_exactly_50_items(cls, v):
        """Validatore che garantisce esattamente NUM_RECOMMENDATIONS elementi."""
        if len(v) != NUM_RECOMMENDATIONS:
            raise ValueError(f"L'array deve contenere esattamente {NUM_RECOMMENDATIONS} elementi, trovati {len(v)}")
        return v

# ----------------------------
# Definizioni Prompt e Parser (MODIFICATO)
# ----------------------------
PROMPT_VARIANTS = {
    "precision_at_k": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n"
        "You are a personal movie recommendation consultant optimizing for PRECISION@K for a specific user. "
        "Your goal is to recommend movies that the user will rate 4 or 5 out of 5. "
        "Carefully analyze the user\\\'s profile and focus on the following elements:\\n"
        "1. Genres that the user has consistently rated highly.\\n"
        "2. Identify key actors, directors, themes, and time periods from the user\\\'s appreciated movies. Prioritize movies in the catalog that share these specific attributes.\\n"
        "3. Actively analyze disliked movies to identify genres, themes, or attributes to avoid.\\n\\n"
        "Precision@k measures how many of the recommended movies will actually be rated positively. "
        "When analyzing the catalog, pay particular attention to:\\n"
        "- Genre matching with positively rated movies.\\n"
        "- Thematic and stylistic similarity to favorite movies.\\n"
        "- Avoid movies similar to those the user did not appreciate.\\n\\n"
        "DO NOT recommend movies based on general popularity or trends, unless these "
        "characteristics align with this specific user\\\'s unique preferences. \\n"
        "<output_requirements>\\n"
        f"1. From the # Movie catalog provided by the user in their message, you MUST select and recommend a list containing EXACTLY {NUM_RECOMMENDATIONS} movie IDs. No more, no less than {NUM_RECOMMENDATIONS}.\\n"
        f"2. The list of {NUM_RECOMMENDATIONS} recommendations MUST be ordered. The first movie ID should be the one you recommend the most (highest probability of positive rating), and the last one the least recommended, based on the user's profile and the provided catalog.\\n"
        f"3. Generating a list with a number of movie IDs different from EXACTLY {NUM_RECOMMENDATIONS} will cause a system error and is strictly forbidden.\\n"
        f"4. Your response MUST include an 'explanation' field (string) detailing the main reasons for your top selections in relation to the user\\\'s profile and the provided movie catalog.\\n"
        "</output_requirements>\\n"
        "<|eot_id|>"
    ),
    "coverage": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n"
        f"You are an expert recommendation system that optimizes for COVERAGE. "
        f"Given a list of movies in the # Movie catalog from the user message, recommend an ORDERED list of EXACTLY {NUM_RECOMMENDATIONS} movies that maximize coverage of different film genres, "
        f"BUT that are still relevant to the specific preferences of the user whose profile you are analyzing. "
        f"Coverage measures the proportion of the entire catalog that the system is able to recommend. "
        f"The goal is to better explore the available movie space and reduce the risk of filter bubbles. "
        f"Make sure your recommendations cover different genres, but are aligned with the user\'s tastes. "
        f"Order the list by putting first the movies that represent a good compromise between genre diversity and user preferences, "
        f"and last those that prioritize pure diversity more at the expense of immediate relevance. "
        f"IMPORTANT: Make specific reference to movies the user has enjoyed to discover related but different genres. "
        f"Each user should receive personalized recommendations based on their unique profile. \\n"
        "<output_requirements>\\n"
        f"1. From the # Movie catalog provided by the user in their message, you MUST select and recommend an ORDERED list of EXACTLY {NUM_RECOMMENDATIONS} movie IDs. This list must maximize genre coverage while remaining relevant to the user's preferences. No more, no less than {NUM_RECOMMENDATIONS} items.\\n"
        f"2. The list of {NUM_RECOMMENDATIONS} recommendations MUST be ordered as described above (compromise between diversity and relevance first, pure diversity last).\\n"
        f"3. It is CRITICAL and MANDATORY that your list contains EXACTLY {NUM_RECOMMENDATIONS} movie IDs. Deviating from this exact number (e.g., providing {NUM_RECOMMENDATIONS-1} or {NUM_RECOMMENDATIONS+1}) will lead to a system failure and is unacceptable.\\n"
        f"4. Your response MUST include an 'explanation' field (string) detailing how your selections achieve genre coverage based on the user's profile and the provided movie catalog.\\n"
        "</output_requirements>\\n"
        "<|eot_id|>"
    )
}

def create_metric_prompt(metric_name: str, metric_description: str) -> PromptTemplate:
    """Crea un PromptTemplate Llama 3.3 formattato per una specifica metrica.
    
    Args:
        metric_name: Il nome della metrica (usato per scopi informativi interni, non nel prompt finale all\'LLM).
        metric_description: Il system prompt Llama 3.3 completo, già formattato con i token
                           <|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>.
    """
    # Il metric_description è il system prompt Llama 3.3 completo (da PROMPT_VARIANTS)
    
    # Contenuto per il messaggio dell'utente
    user_message_content = (
        # Il task specifico è già nel system_prompt (metric_description).
        "# User profile:\\n"
        "{user_profile}\\n\\n"
        "# Movie catalog (use this as the source for your recommendations):\\n"
        "{catalog}\\n\\n"
        "# Required Output Structure (MUST be followed):\\n"
        "<output_format_instructions>\\n"
        f"- The 'recommendations' field MUST be a list of EXACTLY {NUM_RECOMMENDATIONS} integer movie IDs. This count ({NUM_RECOMMENDATIONS}) is absolute, critical, and non-negotiable.\\n"
        f"- The 'recommendations' list MUST be ordered according to the specified metric strategy outlined in the system message.\\n"
        f"- An 'explanation' field (string) detailing the rationale for the {NUM_RECOMMENDATIONS} recommendations MUST be provided.\\n"
        f"- Adherence to providing EXACTLY {NUM_RECOMMENDATIONS} movie IDs is paramount for system functionality. Any deviation will result in failure.\\n"
        "</output_format_instructions>"
    )
    
    # Assembla il template completo per Llama 3.3
    full_prompt_template_str = (
        f"{metric_description}\\n"  # System message (già formattato Llama3)
        "<|start_header_id|>user<|end_header_id|>\\n"
        f"{user_message_content}\\n" # Aggiunto newline alla fine di user_message_content
        "<|eot_id|>\\n"
        "<|start_header_id|>assistant<|end_header_id|>\\n" # Pronto per la risposta dell'LLM
    )
    
    return PromptTemplate(
        input_variables=["catalog", "user_profile"], # Variabili per user_message_content
        template=full_prompt_template_str
    )

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
        
        # Crea il grafo di stati
        workflow = StateGraph(RecommenderState)
        
        # Aggiungi nodi per ogni fase del processo
        # 1. Nodo di inizializzazione
        workflow.add_node("initialize", self._node_initialize)
        
        # 2. Nodo per preparare dati utente
        workflow.add_node("prepare_user_data", self._node_prepare_user_data)
        
        # 3. Nodi per le metriche 
        workflow.add_node("run_precision_metric", self._node_run_precision_metric)
        workflow.add_node("run_coverage_metric", self._node_run_coverage_metric)
        
        # 4. Nodo per raccogliere risultati metriche
        workflow.add_node("collect_metric_results", self._node_collect_metric_results)
        
        # 5. Nodo per passare al prossimo utente o terminare
        workflow.add_node("next_user_or_finish", self._node_next_user_or_finish)
        
        # 6. Nodo valutatore finale
        workflow.add_node("evaluate_all_results", self._node_evaluate_all_results)
        
        # Definire il punto di ingresso
        workflow.set_entry_point("initialize")
        
        # Definire i flussi
        workflow.add_edge("initialize", "prepare_user_data")
        workflow.add_edge("prepare_user_data", "run_precision_metric")
        workflow.add_edge("prepare_user_data", "run_coverage_metric")
        
        # Da run_metric a collect_metric_results (quando entrambi completati)
        workflow.add_conditional_edges(
            "run_precision_metric",
            self._check_metrics_completion,
            {
                "complete": "collect_metric_results",
                "not_complete": "run_precision_metric"  # Self-loop fino al completamento
            }
        )
        workflow.add_conditional_edges(
            "run_coverage_metric",
            self._check_metrics_completion,
            {
                "not_complete": "run_coverage_metric",  # Self-loop fino al completamento
                "complete": "collect_metric_results"
            }
        )
        
        # Dopo aver raccolto i risultati, decidere se passare all'utente successivo o concludere
        workflow.add_conditional_edges(
            "collect_metric_results",
            self._check_users_completion,
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
        # Usa i flag di completamento invece del contatore
        all_completed = state.get("precision_completed", False) and state.get("coverage_completed", False)
        
        if all_completed:
            return "complete"
        return "not_complete"

    def _check_users_completion(self, state: RecommenderState) -> str:
        """Verifica se tutti gli utenti sono stati elaborati."""
        if state["current_user_index"] >= len(state["user_ids"]):
            return "evaluate"
        return "next_user"

    async def _node_initialize(self, state: RecommenderState) -> Dict[str, Any]:
        """Inizializza lo stato del workflow."""
        if not self.datasets_loaded:
            self._load_datasets()
        
        return {
            "user_ids": self.specific_user_ids,
            "current_user_index": 0,
            "all_user_results": {},
            "metric_results": {},
            "precision_at_k_result": None,
            "coverage_result": None,
            "precision_completed": False,
            "coverage_completed": False,
            "metric_tasks_completed": 0,
            "expected_metrics": len(self.current_prompt_variants),
            "held_out_items": {},
            "final_evaluation": None,
            "error": None
        }

    async def _node_prepare_user_data(self, state: RecommenderState) -> Dict[str, Any]:
        """Prepara i dati dell'utente corrente."""
        user_index = state["current_user_index"]
        user_ids = state["user_ids"]
        
        if user_index >= len(user_ids):
            return {"error": "Indice utente fuori range"}
        
        user_id = user_ids[user_index]
        
        # Recupera il profilo utente (stessa logica di run_recommendation_pipeline)
        if user_id not in self.user_profiles.index:
            return {"error": f"Utente {user_id} non trovato."}
            
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
        profile_summary = json.dumps({"user_id": int(user_id), "liked_movies": profile_liked, "disliked_movies": disliked}, ensure_ascii=False)
        
        # Raccogli held_out per questo utente
        held_out = safe_load_list(profile_series.get("held_out_liked_movies"))
        updated_held_out = state["held_out_items"].copy()
        updated_held_out[user_id] = held_out
        
        print(f"\n--- Raccomandazioni per utente {user_id} --- (Profilo con {len(profile_liked)} liked, {len(disliked)} disliked. Held-out: {len(held_out)} items)")
        
        # Genera cataloghi specifici per metrica usando RAG
        catalog_precision = None
        catalog_coverage = None
        catalog_json_fallback = self.get_optimized_catalog(limit=300)
        
        if self.rag:
            try:
                # Catalogo per Precision@k
                print(f"RAG: Tentativo chiamata similarity_search per precision_at_k per utente {user_id}...")
                start_rag_p = time.time()
                cat_p = self.rag.similarity_search(profile_summary, k=300, metric_focus="precision_at_k", user_id=int(user_id))
                end_rag_p = time.time()
                print(f"RAG: Tempo impiegato per precision_at_k: {end_rag_p - start_rag_p:.2f} secondi")
                catalog_precision = json.dumps(cat_p[:300], ensure_ascii=False)
                print(f"RAG: Generato catalogo specifico per precision_at_k (size: {len(cat_p)}) (Successo)")
            except Exception as e:
                print(f"Errore RAG (precision_at_k) user {user_id}: {e}. Uso catalogo fallback.")
                catalog_precision = catalog_json_fallback

            try:
                # Catalogo per Coverage
                coverage_query = "diversi generi film non ancora visti dall'utente" + profile_summary 
                print(f"RAG: Tentativo chiamata similarity_search per coverage per utente {user_id}...")
                start_rag_c = time.time()
                cat_c = self.rag.similarity_search(coverage_query, k=300, metric_focus="coverage", user_id=int(user_id))
                end_rag_c = time.time()
                print(f"RAG: Tempo impiegato per coverage: {end_rag_c - start_rag_c:.2f} secondi")
                catalog_coverage = json.dumps(cat_c[:300], ensure_ascii=False)
                print(f"RAG: Generato catalogo specifico per coverage (size: {len(cat_c)}) (Successo)")
            except Exception as e:
                print(f"Errore RAG (coverage) user {user_id}: {e}. Uso catalogo fallback.")
                catalog_coverage = catalog_json_fallback
        else:
            print("RAG non disponibile. Uso catalogo fallback per tutte le metriche.")
            catalog_precision = catalog_coverage = catalog_json_fallback
        
        return {
            "user_id": user_id,
            "user_profile": profile_summary,
            "catalog_precision": catalog_precision,
            "catalog_coverage": catalog_coverage,
            "precision_completed": False,
            "coverage_completed": False,
            "metric_tasks_completed": 0,  # Resetta per il nuovo utente
            "metric_results": {},  # Resetta per il nuovo utente
            "held_out_items": updated_held_out
        }

    async def _node_run_precision_metric(self, state: RecommenderState) -> Dict[str, Any]:
        """Esegue la logica per la metrica precision_at_k."""
        user_id_for_log = state.get('user_id') # Prendi user_id dallo stato
        print(f"Executing precision_at_k metric for user {user_id_for_log}..." )
        
        metric_name = "precision_at_k"
        catalog = state["catalog_precision"]
        user_profile = state["user_profile"]
        
        # Riutilizza la stessa logica del tool esistente, passando user_id
        result = await self._run_metric_tool_internal(metric_name, catalog, user_profile, user_id=user_id_for_log)
        
        # Usa precision_at_k_result invece di metric_results
        return {
            "precision_at_k_result": result,
            "precision_completed": True
        }

    async def _node_run_coverage_metric(self, state: RecommenderState) -> Dict[str, Any]:
        """Esegue la logica per la metrica coverage."""
        user_id_for_log = state.get('user_id') # Prendi user_id dallo stato
        print(f"Executing coverage metric for user {user_id_for_log}...")
        
        metric_name = "coverage"
        catalog = state["catalog_coverage"]
        user_profile = state["user_profile"]
        
        # Riutilizza la stessa logica del tool esistente, passando user_id
        result = await self._run_metric_tool_internal(metric_name, catalog, user_profile, user_id=user_id_for_log)
        
        # Usa coverage_result invece di metric_results
        return {
            "coverage_result": result,
            "coverage_completed": True
        }

    async def _node_collect_metric_results(self, state: RecommenderState) -> Dict[str, Any]:
        """Raccoglie i risultati delle metriche per l'utente corrente."""
        user_id = state["user_id"]
        
        # Combina i risultati dalle metriche separate in metric_results
        precision_result = state.get("precision_at_k_result", {})
        coverage_result = state.get("coverage_result", {})
        
        combined_results = {}
        if precision_result:
            combined_results["precision_at_k"] = precision_result
        if coverage_result:
            combined_results["coverage"] = coverage_result
        
        # Calcola il numero di metriche completate dai flag
        completed_count = 0
        if state.get("precision_completed", False):
            completed_count += 1
        if state.get("coverage_completed", False):
            completed_count += 1
        
        print(f"Raccolti risultati metriche per utente {user_id}: {', '.join(combined_results.keys())}")
        
        # Aggiorna i risultati complessivi per utente
        all_results = state["all_user_results"].copy()
        all_results[user_id] = combined_results
        
        # Passa all'utente successivo
        return {
            "all_user_results": all_results,
            "current_user_index": state["current_user_index"] + 1,
            "metric_results": combined_results,  # Aggiorna anche metric_results per compatibilità 
            "metric_tasks_completed": completed_count,  # Aggiorna direttamente qui
            
            # Reset dei risultati delle singole metriche per il prossimo utente
            "precision_at_k_result": None,
            "coverage_result": None,
            "precision_completed": False,
            "coverage_completed": False
        }

    async def _node_next_user_or_finish(self, state: RecommenderState) -> Dict[str, Any]:
        """Determina se passare all'utente successivo o terminare."""
        # Questo nodo non è più necessario con i conditional_edges ma lo mantengo per chiarezza
        return {}

    async def _node_evaluate_all_results(self, state: RecommenderState) -> Dict[str, Any]:
        """Valuta i risultati di tutti gli utenti."""
        all_results = state["all_user_results"]
        
        # Converti in formato JSON per l'evaluator
        all_results_str = json.dumps(all_results, ensure_ascii=False, indent=2)
        eval_catalog = self.get_optimized_catalog(limit=300)
        
        print("\n--- Valutazione Aggregata ---")
        start_eval = time.time()
        
        # Chiama la stessa logica dell'evaluator esistente
        final_evaluation = await self._evaluate_recommendations_internal(
            all_recommendations_str=all_results_str,
            catalog_str=eval_catalog
        )
        
        end_eval = time.time()
        print(f"Tempo impiegato per valutazione aggregata: {end_eval - start_eval:.2f} secondi")
        print(f"Final recommendations: {final_evaluation.get('final_recommendations', [])}")
        
        return {
            "final_evaluation": final_evaluation
        }

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
        
        # Usa la versione Llama 3.3 del prompt
        eval_system_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n"
            "You are an expert recommendation aggregator. Your task is to analyze recommendations from different metrics "
            f"and create an OPTIMAL final recommendation list of EXACTLY {NUM_RECOMMENDATIONS} movies. "
            "Balance the precision@k and coverage metrics, giving more weight to precision for top recommendations while "
            "ensuring adequate coverage of movie genres."
            "\\n\\n"
            "RULES:\\n"
            "1. Balance both metrics, but precision@k should be the primary consideration for top recommendations.\\n"
            "2. Consider the ordering of movies within each metric's list - higher ranked items are more important.\\n"
            "3. Provide a justification that explains your aggregation logic and the trade-offs between metrics.\\n"
            "4. Ensure your final recommendations form a cohesive, personalized set for the user.\\n"
            f"5. STRICT REQUIREMENT: Return EXACTLY {NUM_RECOMMENDATIONS} movie IDs, no more and no less. Your output MUST BE a valid JSON object matching the Pydantic schema.\\n"
            "\\n"
            "<|eot_id|>"
        )
        
        # Template per il messaggio utente  
        user_template_str = (
            "<|start_header_id|>user<|end_header_id|>\\n"
            "# Recommendations per Metric:\\n"
            "{all_recommendations}\\n\\n"
            "# Movie catalog for reference:\\n"
            "{catalog}\\n\\n"
            "# Required Output Format:\\n"
            f"You MUST provide EXACTLY {NUM_RECOMMENDATIONS} movie IDs in your final_recommendations list. "
            "This is a strict requirement - more or fewer IDs will cause a system error. "
            "Include detailed justification and trade-off analysis. Your entire output must be a single, valid JSON object."
            "{feedback_block}" # Placeholder per il feedback
            "\\n<|eot_id|>\\n"
            "<|start_header_id|>assistant<|end_header_id|>\\n"
        )
        
        # Crea il template completo iniziale
        full_eval_prompt_template = PromptTemplate(
            input_variables=["all_recommendations", "catalog", "feedback_block"], # Aggiunto feedback_block
            template=f"{eval_system_prompt}\\n{user_template_str}"
        )
        
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
    
    # Questo metodo è mantenuto per compatibilità ma non viene più usato
    def _build_metric_tools(self) -> List[Tool]:
        """Costruisce i Tools per ciascuna metrica di raccomandazione. Mantenuto per compatibilità."""
        print("AVVISO: _build_metric_tools è mantenuto per compatibilità ma non è utilizzato da LangGraph.")
        # Resto del codice originale...
        
    # Questo metodo è mantenuto per compatibilità ma non viene più usato
    def _build_evaluator_tool(self) -> Tool:
        """Costruisce il Tool per valutare e combinare le raccomandazioni. Mantenuto per compatibilità."""
        print("AVVISO: _build_evaluator_tool è mantenuto per compatibilità ma non è utilizzato da LangGraph.")
        # Resto del codice originale...
    
    # Questo metodo non viene più usato, sostituito da _initialize_langgraph
    def _initialize_agent(self) -> None:
        """DEPRECATO: Usa _initialize_langgraph() invece."""
        print("AVVISO: _initialize_agent è deprecato. Utilizzare _initialize_langgraph() invece.")

    def initialize_system(self, force_reload_data: bool = False, force_recreate_vector_store: bool = False) -> None:
        """Metodo pubblico per inizializzare o reinizializzare il sistema."""
        print("\n=== Inizializzazione Sistema ===")
        self._load_datasets(force_reload=force_reload_data)
        self._initialize_rag(force_recreate_vector_store=force_recreate_vector_store)
        
        # MODIFICATO: usa _initialize_langgraph invece di _initialize_agent
        self._initialize_langgraph()
        
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