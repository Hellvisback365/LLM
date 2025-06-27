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
    "temperature": 0.2,  # Leggermente aumentato per più creatività ma ancora consistente
    "max_tokens": 4000,  # Aumentato per dare più spazio all'LLM per completare le risposte
}


LLM_MODEL_ID = "meta-llama/llama-3.2-3b-instruct"

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
    
    def __init__(self, specific_user_ids: List[int] = None, model_id: str = LLM_MODEL_ID):
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

                if not force_reload and all(os.path.exists(f) for f in [profiles_file, movies_file]):
                    print("Caricamento dati da file elaborati...")
                    # self.filtered_ratings = pd.read_csv(ratings_file)
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
                    # self.filtered_ratings.to_csv(ratings_file, index=False)
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
        
        # 5. Nodo per aggregazione per-utente
        workflow.add_node("aggregate_user_recommendations", self.graph_nodes.node_aggregate_user_recommendations)
        
        # 6. Nodo per passare al prossimo utente o terminare
        workflow.add_node("next_user_or_finish", self.graph_nodes.node_next_user_or_finish)
        
        # 7. Nodo valutatore finale
        workflow.add_node("evaluate_all_results", self.graph_nodes.node_evaluate_all_results)
        
        # Definire il punto di ingresso
        workflow.set_entry_point("initialize")
        
        # Definire i flussi - usa un approccio sequenziale invece di parallelo per evitare problemi di sincronizzazione
        workflow.add_edge("initialize", "prepare_user_data")
        workflow.add_edge("prepare_user_data", "run_precision_metric")
        workflow.add_edge("run_precision_metric", "run_coverage_metric")
        workflow.add_edge("run_coverage_metric", "collect_metric_results")
        
        # Dopo aver raccolto i risultati, decidere se passare all'utente successivo o concludere
        workflow.add_conditional_edges(
            "collect_metric_results",
            self.graph_nodes.check_users_completion,
            {
                "aggregate_user": "aggregate_user_recommendations",  # Prima aggrega l'utente
                "evaluate": "evaluate_all_results"  # Tutti gli utenti elaborati, valuta
            }
        )
        
        # Dopo l'aggregazione per-utente, passa al prossimo utente
        workflow.add_edge("aggregate_user_recommendations", "prepare_user_data")
        
        # L'evaluator conclude il grafo
        workflow.add_edge("evaluate_all_results", END)
        
        # Compila il grafo con un limite di ricorsione più alto
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
        
        print(f"DEBUG _run_metric_tool_internal: Called for metric '{metric_name}', user {user_id}") 
        metric_desc = self.current_prompt_variants.get(metric_name)
        print(f"DEBUG _run_metric_tool_internal: metric_desc = {repr(metric_desc)}") 
        
        prompt_template = create_metric_prompt(metric_name, metric_desc)

        for attempt in range(max_attempts):
            try: 
                prompt_str = None 
                try: 
                    print(f"DEBUG _run_metric_tool_internal: Attempting prompt_template.format() for metric '{metric_name}', user {user_id}, attempt {attempt + 1}") 
                    prompt_str = prompt_template.format(catalog=catalog, user_profile=user_profile)
                except Exception as format_exception:
                    print(f"!!! ERROR during prompt_template.format() for metric '{metric_name}', user {user_id}, attempt {attempt + 1} !!!")
                    print(f"  Exception type: {type(format_exception)}")
                    raise format_exception 

                print(f"Attempt {attempt+1}/{max_attempts} invoking LLM for metric: {metric_name} (user {user_id if user_id is not None else 'N/A'}).")
                
                # Usa il normale llm.ainvoke invece di structured_output per gestire meglio il markdown
                response = await self.llm.ainvoke(prompt_str)
                raw_content = getattr(response, 'content', str(response))
                
                # Pulisci la risposta rimuovendo markdown e altri artefatti
                cleaned_content = self._clean_llm_json_response(raw_content)
                
                # Parsing manuale del JSON con gestione robusta
                import json
                try:
                    parsed_json = json.loads(cleaned_content)
                    
                    # Gestione flessibile delle raccomandazioni
                    recommendations = parsed_json.get("recommendations", [])
                    explanation = parsed_json.get("explanation", "No explanation provided")
                    
                    # Assicurati che recommendations sia una lista di interi
                    if not isinstance(recommendations, list):
                        print(f"DEBUG metric {metric_name} (user {user_id if user_id is not None else 'N/A'}, attempt {attempt+1}): recommendations non è una lista, generando fallback")
                        return self._generate_fallback_recommendation(metric_name, user_id)
                    
                    # Converti tutti gli elementi in interi e filtra quelli non validi
                    valid_recommendations = []
                    for item in recommendations:
                        try:
                            valid_recommendations.append(int(item))
                        except (ValueError, TypeError):
                            continue
                    
                    # Aggiusta la lunghezza della lista: tronca se troppo lunga, riempi se troppo corta
                    target_length = 20  # NUM_RECOMMENDATIONS
                    if len(valid_recommendations) > target_length:
                        valid_recommendations = valid_recommendations[:target_length]
                        print(f"DEBUG metric {metric_name} (user {user_id if user_id is not None else 'N/A'}, attempt {attempt+1}): Troncata lista da {len(recommendations)} a {target_length}")
                    elif len(valid_recommendations) < target_length:
                        # Riempi con film casuali dal catalogo
                        try:
                            catalog_data = json.loads(catalog)
                            if isinstance(catalog_data, list) and catalog_data:
                                available_ids = [movie.get('movie_id') for movie in catalog_data if movie.get('movie_id')]
                                available_ids = [int(id) for id in available_ids if id is not None]
                                # Rimuovi IDs già presenti
                                available_ids = [id for id in available_ids if id not in valid_recommendations]
                                
                                import random
                                needed = target_length - len(valid_recommendations)
                                if len(available_ids) >= needed:
                                    valid_recommendations.extend(random.sample(available_ids, needed))
                                else:
                                    valid_recommendations.extend(available_ids)
                                    # Se ancora non basta, ripeti gli esistenti
                                    while len(valid_recommendations) < target_length:
                                        valid_recommendations.append(valid_recommendations[0] if valid_recommendations else 1)
                                
                                print(f"DEBUG metric {metric_name} (user {user_id if user_id is not None else 'N/A'}, attempt {attempt+1}): Estesa lista da {len(recommendations)} a {len(valid_recommendations)}")
                        except Exception as fill_error:
                            print(f"DEBUG metric {metric_name} (user {user_id if user_id is not None else 'N/A'}, attempt {attempt+1}): Errore nel riempimento lista: {fill_error}")
                            return self._generate_fallback_recommendation(metric_name, user_id)
                    
                    # Valida usando il schema Pydantic con la lista corretta
                    final_data = {
                        "recommendations": valid_recommendations,
                        "explanation": explanation
                    }
                    recommendation_obj = RecommendationOutput(**final_data)
                    print(f"DEBUG metric {metric_name} (user {user_id if user_id is not None else 'N/A'}, attempt {attempt+1}): JSON parsing e validazione successful con {len(valid_recommendations)} raccomandazioni.")
                    return {"metric": metric_name, **recommendation_obj.dict()}
                except json.JSONDecodeError as e:
                    print(f"DEBUG metric {metric_name} (user {user_id if user_id is not None else 'N/A'}, attempt {attempt+1}): JSON decode error - {str(e)[:300]}...")
                    if attempt == max_attempts - 1:
                        print(f"Generating fallback response for metric {metric_name} (user {user_id}) after JSON decode failures")
                        return self._generate_fallback_recommendation(metric_name, user_id)
                    continue
                
            except ValidationError as e:
                user_id_str = str(user_id) if user_id is not None else 'N/A'
                error_message_raw = str(e)
                print(f"DEBUG metric {metric_name} (user {user_id_str}, attempt {attempt+1}): Pydantic ValidationError - {error_message_raw[:500]}...")
                
                # Se è l'ultimo tentativo, restituisci un fallback
                if attempt == max_attempts - 1:
                    print(f"Generating fallback response for metric {metric_name} (user {user_id_str}) after validation failures")
                    return self._generate_fallback_recommendation(metric_name, user_id)
                
                # Altrimenti continua al prossimo tentativo
                continue
            
            except RateLimitError as e:
                print(f"Rate limit error in attempt {attempt+1} for metric {metric_name} (user {user_id if user_id is not None else 'N/A'}): {e}. Retrying...")
                if attempt < max_attempts - 1:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue 
                else:
                    print(f"All attempts failed for {metric_name} (user {user_id if user_id is not None else 'N/A'}) due to RateLimitError: {e}")
                    return self._generate_fallback_recommendation(metric_name, user_id)

            except Exception as e: 
                user_id_str = str(user_id) if user_id is not None else 'N/A'
                error_message_raw = str(e)
                print(f"DEBUG metric {metric_name} (user {user_id_str}, attempt {attempt+1}): General Exception - {type(e).__name__}: {error_message_raw[:500]}...")
                
                # Se è l'ultimo tentativo, restituisci un fallback
                if attempt == max_attempts - 1:
                    print(f"Generating fallback response for metric {metric_name} (user {user_id_str}) after general errors")
                    return self._generate_fallback_recommendation(metric_name, user_id)
                
                # Altrimenti continua al prossimo tentativo
                continue

        # Fallback finale se tutti i tentativi falliscono
        return self._generate_fallback_recommendation(metric_name, user_id)

    def _generate_fallback_recommendation(self, metric_name: str, user_id: Optional[int] = None) -> Dict:
        """Genera una raccomandazione di fallback quando l'LLM fallisce."""
        import random
        
        # Genera raccomandazioni casuali dai film disponibili
        if self.movies is not None and not self.movies.empty:
            movie_ids = self.movies['movie_id'].tolist()
            fallback_recommendations = random.sample(movie_ids, min(NUM_RECOMMENDATIONS, len(movie_ids)))
        else:
            # Fallback estremo con ID casuali
            fallback_recommendations = list(range(1, NUM_RECOMMENDATIONS + 1))
        
        return {
            "metric": metric_name,
            "recommendations": fallback_recommendations,
            "explanation": f"Fallback recommendations generated due to LLM errors for metric {metric_name} (user {user_id}). These are random selections and should not be used for evaluation."
        }

    async def _evaluate_recommendations_internal(self, all_recommendations_str: str, catalog_str: str) -> Dict:
        """Versione interna di evaluate_recommendations_tool che può essere chiamata dai nodi."""
        max_attempts = 3
        
        print(f"DEBUG _evaluate_recommendations_internal: Called.") 
        prompt_template = create_evaluation_prompt()

        for attempt in range(max_attempts):
            try: 
                print(f"DEBUG _evaluate_recommendations_internal: Attempting prompt format, attempt {attempt + 1}") 
                prompt_str = prompt_template.format(
                    all_recommendations=all_recommendations_str,
                    catalog=catalog_str,
                    feedback_block=""
                )
                
                print(f"Attempt {attempt+1}/{max_attempts} invoking LLM for evaluation.")
                
                # Usa llm.ainvoke standard e pulizia manuale del JSON per gestire output con markdown
                response = await self.llm.ainvoke(prompt_str)
                raw_content = getattr(response, 'content', str(response))
                
                cleaned_content = self._clean_llm_json_response(raw_content)
                
                import json
                try:
                    parsed_json = json.loads(cleaned_content)
                    evaluation_obj = EvaluationOutput(**parsed_json)
                    print(f"DEBUG evaluation (attempt {attempt+1}): JSON parsing and validation successful.")
                    return evaluation_obj.dict()
                except json.JSONDecodeError as e:
                    print(f"DEBUG evaluation (attempt {attempt+1}): JSON decode error - {str(e)[:300]}...")
                    if attempt == max_attempts - 1:
                        print(f"Generating fallback evaluation after JSON decode failures")
                        return self._generate_fallback_evaluation()
                    continue
            
            except ValidationError as e:
                error_message_raw = str(e)
                print(f"DEBUG evaluation (attempt {attempt+1}): Pydantic ValidationError - {error_message_raw[:500]}...")
                
                # Se è l'ultimo tentativo, restituisci un fallback
                if attempt == max_attempts - 1:
                    print(f"Generating fallback evaluation after validation failures")
                    return self._generate_fallback_evaluation()
                
                continue

            except RateLimitError as e:
                print(f"Rate limit error in evaluation attempt {attempt+1}: {e}. Retrying...")
                if attempt < max_attempts - 1:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    print(f"All evaluation attempts failed due to RateLimitError after {max_attempts} attempts: {e}")
                    return self._generate_fallback_evaluation()

            except Exception as e:
                error_message_raw = str(e)
                print(f"DEBUG evaluation (attempt {attempt+1}): General Exception - {type(e).__name__}: {error_message_raw[:500]}...")
                
                # Se è l'ultimo tentativo, restituisci un fallback
                if attempt == max_attempts - 1:
                    print(f"Generating fallback evaluation after general errors")
                    return self._generate_fallback_evaluation()
                
                continue

        # Fallback finale se tutti i tentativi falliscono
        return self._generate_fallback_evaluation()

    def _generate_fallback_evaluation(self) -> Dict:
        """Genera una valutazione di fallback quando l'LLM fallisce."""
        import random
        
        # Genera raccomandazioni casuali dai film disponibili
        if self.movies is not None and not self.movies.empty:
            movie_ids = self.movies['movie_id'].tolist()
            fallback_recommendations = random.sample(movie_ids, min(NUM_RECOMMENDATIONS, len(movie_ids)))
        else:
            # Fallback estremo con ID casuali
            fallback_recommendations = list(range(1, NUM_RECOMMENDATIONS + 1))
        
        return {
            "final_recommendations": fallback_recommendations,
            "justification": "Fallback evaluation generated due to LLM errors. These are random selections and should not be used for actual evaluation.",
            "trade_offs": "N/A - Fallback response due to system errors"
        }

    async def evaluate_final_recommendations(self, all_recommendations: Dict, catalog: str) -> Dict:
        """
        Aggrega e valuta le raccomandazioni finali da metriche multiple per un singolo utente
        utilizzando l'intelligenza dell'LLM per bilanciare precision@k e coverage.
        
        Args:
            all_recommendations: Dizionario con le raccomandazioni per diverse metriche
                                Formato: {"precision_at_k": {"user_X": {...}}, "coverage": {"user_X": {...}}}
            catalog: Catalogo dei film in formato JSON string
            
        Returns:
            Dict con le raccomandazioni finali aggregate, giustificazione e trade-offs
        """
        try:
            # Estrai le raccomandazioni per ciascuna metrica
            precision_data = all_recommendations.get("precision_at_k", {})
            coverage_data = all_recommendations.get("coverage", {})
            
            # Estrai le liste di raccomandazioni e spiegazioni
            precision_recs = []
            coverage_recs = []
            precision_explanation = ""
            coverage_explanation = ""
            
            for user_data in precision_data.values():
                precision_recs.extend(user_data.get("recommendations", []))
                precision_explanation = user_data.get("explanation", "")
            
            for user_data in coverage_data.values():
                coverage_recs.extend(user_data.get("recommendations", []))
                coverage_explanation = user_data.get("explanation", "")
            
            # Chiama l'LLM per bilanciare intelligentemente le metriche
            llm_result = await self._call_llm_aggregator(
                precision_recs, coverage_recs, 
                precision_explanation, coverage_explanation, 
                catalog
            )
            
            if llm_result and llm_result.get("recommendations"):
                return {
                    "final_recommendations": llm_result["recommendations"],
                    "justification": llm_result.get("justification", ""),
                    "trade_offs": llm_result.get("trade_offs", "")
                }
            else:
                # Fallback alla logica precedente in caso di errore LLM
                print("WARNING: LLM aggregation failed, using fallback logic")
                return await self._fallback_aggregation(precision_recs, coverage_recs)
            
        except Exception as e:
            print(f"Error in evaluate_final_recommendations: {e}")
            # Fallback alla logica precedente in caso di errore
            return await self._fallback_aggregation(precision_recs if 'precision_recs' in locals() else [], 
                                                   coverage_recs if 'coverage_recs' in locals() else [])

    async def _call_llm_aggregator(self, precision_recs: List[int], coverage_recs: List[int], 
                                   precision_explanation: str, coverage_explanation: str, 
                                   catalog: str) -> Dict:
        """
        Chiama l'LLM per aggregare intelligentemente le raccomandazioni precision@k e coverage.
        
        Args:
            precision_recs: Lista delle raccomandazioni precision@k
            coverage_recs: Lista delle raccomandazioni coverage  
            precision_explanation: Spiegazione delle raccomandazioni precision@k
            coverage_explanation: Spiegazione delle raccomandazioni coverage
            catalog: Catalogo dei film
            
        Returns:
            Dict con raccomandazioni aggregate, giustificazione e trade-offs
        """
        
        # Crea il prompt per l'LLM aggregatore
        aggregator_prompt = self._create_aggregator_prompt(
            precision_recs, coverage_recs, 
            precision_explanation, coverage_explanation
        )
        
        try:
            # Chiama l'LLM con il prompt
            response = await self._call_llm_with_aggregator_prompt(aggregator_prompt)
            
            # Parsa la risposta
            parsed_result = self._parse_aggregator_response(response)
            
            return parsed_result
            
        except Exception as e:
            print(f"ERROR in LLM aggregator: {e}")
            return None

    def _create_aggregator_prompt(self, precision_recs: List[int], coverage_recs: List[int],
                                  precision_explanation: str, coverage_explanation: str) -> str:
        """
        Crea un prompt efficace per l'LLM aggregatore.
        """
        
        # Trova sovrapposizioni per fornire più contesto
        overlap = set(precision_recs) & set(coverage_recs)
        precision_only = [r for r in precision_recs if r not in coverage_recs]
        coverage_only = [r for r in coverage_recs if r not in precision_recs]
        
        prompt = f"""
Tu sei un esperto sistema di raccomandazione cinematografico. Devi creare una lista finale di 20 raccomandazioni 
bilanciando intelligentemente due metriche complementari:

**PRECISION@K RECOMMENDATIONS** (Rilevanza personalizzata):
- Film: {precision_recs}
- Spiegazione: {precision_explanation}
- Solo in precision@k: {precision_only} ({len(precision_only)} film)

**COVERAGE RECOMMENDATIONS** (Diversità ed esplorazione):
- Film: {coverage_recs}
- Spiegazione: {coverage_explanation}  
- Solo in coverage: {coverage_only} ({len(coverage_only)} film)

**SOVRAPPOSIZIONI**: {list(overlap)} ({len(overlap)} film presenti in entrambe)

**COMPITO:**
Analizza queste due liste e crea una strategia di bilanciamento ottimale. NON seguire proporzioni fisse, 
ma usa la tua intelligenza per valutare:

1. **Qualità delle raccomandazioni**: Quale lista sembra più robusta/rilevante?
2. **Diversità necessaria**: L'utente ha bisogno di più esplorazione o più precision?
3. **Sovrapposizioni**: Come sfruttare i film che appaiono in entrambe?
4. **Complementarità**: Come combinare al meglio rilevanza e diversità?

**OUTPUT RICHIESTO** (JSON):
{{
    "recommendations": [lista di esattamente 20 ID film],
    "strategy_used": "breve descrizione della strategia scelta",
    "precision_weight": numero_da_0_a_1,
    "coverage_weight": numero_da_0_a_1,
    "justification": "spiegazione dettagliata delle scelte fatte",
    "trade_offs": "analisi dei trade-off tra precision e coverage"
}}

**VINCOLI:**
- Esattamente 20 raccomandazioni finali
- Usa SOLO film dalle liste fornite
- NO duplicati
- Spiega la tua logica di bilanciamento
- Considera il contesto specifico di questo utente

Crea la migliore strategia di bilanciamento basata sulla tua analisi!
"""
        
        return prompt.strip()

    async def _call_llm_with_aggregator_prompt(self, prompt: str) -> str:
        """
        Chiama l'LLM con il prompt dell'aggregatore.
        """
        try:
            messages = [
                {"role": "system", "content": "You are an expert movie recommendation aggregator."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            print(f"ERROR calling LLM aggregator: {e}")
            raise e

    def _parse_aggregator_response(self, response: str) -> Dict:
        """
        Parsa la risposta dell'LLM aggregatore con debugging migliorato.
        """
        try:
            import json
            import re
            
            print(f"DEBUG: Raw LLM response length: {len(response)}")
            print(f"DEBUG: Raw LLM response FULL:")
            print("="*50)
            print(response)
            print("="*50)
            
            # Prova diversi pattern per estrarre JSON
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Pattern complesso originale
                r'\{.*?\}',  # Pattern semplice
                r'```json\s*(\{.*?\})\s*```',  # JSON in code block
                r'```\s*(\{.*?\})\s*```',  # JSON in generic code block
            ]
            
            for i, pattern in enumerate(json_patterns):
                json_matches = re.findall(pattern, response, re.DOTALL)
                print(f"DEBUG: Pattern {i+1} found {len(json_matches)} matches")
                
                for j, json_str in enumerate(json_matches):
                    try:
                        print(f"DEBUG: Trying to parse match {j+1}: {json_str[:200]}...")
                        result = json.loads(json_str)
                        
                        # Valida la struttura
                        if ("recommendations" in result and 
                            isinstance(result["recommendations"], list) and
                            len(result["recommendations"]) >= 15):  # Relaxed validation
                            
                            recommendations = result["recommendations"][:NUM_RECOMMENDATIONS]
                            print(f"DEBUG: Successfully parsed JSON with {len(recommendations)} recommendations")
                            
                            return {
                                "recommendations": recommendations,
                                "justification": result.get("justification", "LLM-based intelligent aggregation"),
                                "trade_offs": result.get("trade_offs", "Balanced precision and coverage using AI analysis"),
                                "strategy_used": result.get("strategy_used", "AI-optimized"),
                                "precision_weight": result.get("precision_weight", 0.7),
                                "coverage_weight": result.get("coverage_weight", 0.3)
                            }
                            
                    except json.JSONDecodeError as parse_error:
                        print(f"DEBUG: JSON parse error for match {j+1}: {parse_error}")
                        continue
                        
            # Se il parsing fallisce, prova a estrarre solo la lista
            print("WARNING: Could not parse full JSON, attempting list extraction")
            return self._extract_recommendations_from_text(response)
            
        except Exception as e:
            print(f"ERROR parsing aggregator response: {e}")
            return None

    def _extract_recommendations_from_text(self, text: str) -> Dict:
        """
        Estrae le raccomandazioni dal testo quando il JSON non è parsabile.
        """
        try:
            import re
            
            # Cerca liste di numeri
            numbers = re.findall(r'\d+', text)
            if len(numbers) >= NUM_RECOMMENDATIONS:
                recommendations = [int(n) for n in numbers[:NUM_RECOMMENDATIONS]]
                
                return {
                    "recommendations": recommendations,
                    "justification": "LLM-based aggregation (extracted from text)",
                    "trade_offs": "Intelligent balance between precision and coverage"
                }
            
            return None
            
        except Exception as e:
            print(f"ERROR extracting recommendations from text: {e}")
            return None

    async def _fallback_aggregation(self, precision_recs: List[int], coverage_recs: List[int]) -> Dict:
        """
        Aggregazione di fallback quando l'LLM fallisce.
        """
        print("Using fallback aggregation logic")
        
        seen = set()
        final_recs = []
        
        # Strategia di fallback semplice: 70/30
        precision_count = int(NUM_RECOMMENDATIONS * 0.7)
        coverage_count = NUM_RECOMMENDATIONS - precision_count
        
        # Aggiungi precision
        for rec in precision_recs[:precision_count]:
            if rec not in seen:
                final_recs.append(rec)
                seen.add(rec)
        
        # Aggiungi coverage
        coverage_added = 0
        for rec in coverage_recs:
            if rec not in seen and coverage_added < coverage_count:
                final_recs.append(rec)
                seen.add(rec)
                coverage_added += 1
        
        # Completa se necessario
        for rec in precision_recs[precision_count:]:
            if rec not in seen and len(final_recs) < NUM_RECOMMENDATIONS:
                final_recs.append(rec)
                seen.add(rec)
        
        # Se ancora mancano, aggiungi film casuali
        if len(final_recs) < NUM_RECOMMENDATIONS and self.movies is not None:
            available_movies = self.movies['movie_id'].tolist()
            import random
            additional_recs = [mid for mid in available_movies if mid not in seen]
            if additional_recs:
                needed = NUM_RECOMMENDATIONS - len(final_recs)
                final_recs.extend(random.sample(additional_recs, min(needed, len(additional_recs))))
        
        return {
            "final_recommendations": final_recs[:NUM_RECOMMENDATIONS],
            "justification": "Fallback aggregation using 70/30 precision/coverage split",
            "trade_offs": "Default balance when LLM aggregation is unavailable"
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

    def get_optimized_catalog(self, limit: int = 50) -> str:
        """Ottiene il catalogo ottimizzato per l'LLM con limite molto ridotto per stabilità."""
        catalog_path = os.path.join('data', 'processed', 'optimized_catalog.json')
        try:
            if os.path.exists(catalog_path):
                with open(catalog_path, 'r', encoding='utf-8') as f: 
                    catalog_data = json.load(f)
                # Riduce drasticamente il catalogo per evitare errori LLM
                limited_data = catalog_data[:limit] if limit else catalog_data
                return json.dumps(limited_data, ensure_ascii=False, separators=(',', ':'))
            elif self.rag:
                 print("Catalogo ottimizzato non trovato, genero da RAG...")
                 if self.movies is None or self.movies.empty: self._load_datasets()
                 movies_list = self.movies.to_dict('records')
                 return self.rag.get_optimized_catalog_for_llm(movies_list, limit=limit)
            else: 
                print("Attenzione: RAG non inizializzato.")
                # Fallback a un catalogo molto piccolo
                if self.movies is not None and not self.movies.empty:
                    fallback_movies = self.movies.head(30).to_dict('records')
                    return json.dumps(fallback_movies, ensure_ascii=False, separators=(',', ':'))
                return "[]"
        except Exception as e: 
            print(f"Errore get_optimized_catalog: {e}")
            return "[]"
             
    async def run_recommendation_pipeline(self, use_prompt_variants: Dict = None, experiment_name: Optional[str] = None, batch_size: int = 50) -> Tuple[Dict, Dict, Dict[int, List[int]]]:
        """
        Esegue l'intera pipeline di raccomandazione per tutti gli utenti specificati
        utilizzando LangGraph per l'orchestrazione con processamento a batch.
        Args:
            use_prompt_variants: Dizionario di prompt da usare per questa run.
            experiment_name: Nome dell'esperimento, se applicabile.
            batch_size: Numero di utenti da processare per ogni batch (default: 50).
        """
        # MODIFICATO: verifica che LangGraph sia inizializzato anziché l'agente
        if not self.recommender_graph:
            raise RuntimeError("LangGraph non inizializzato. Chiamare initialize_system() prima.")
        
        if not self.datasets_loaded or self.user_profiles is None or self.user_profiles.empty:
            raise RuntimeError("Dataset non caricati o profili utente vuoti. Chiamare initialize_system() prima.")

        # Imposta le varianti di prompt da usare per questa run
        self.current_prompt_variants = use_prompt_variants if use_prompt_variants is not None else PROMPT_VARIANTS.copy()
        
        start_all_users = time.time()
        
        # Recupera tutti gli utenti da processare
        all_user_ids = self.specific_user_ids if self.specific_user_ids else list(self.user_profiles.index)
        total_users = len(all_user_ids)
        
        print(f"\n=== Avvio pipeline con LangGraph (BATCH MODE) ===")
        print(f"Utenti totali: {total_users}")
        print(f"Dimensione batch: {batch_size}")
        
        # Carica checkpoint se disponibile
        last_batch, all_user_metric_results, all_held_out_items = self._load_latest_checkpoint()
        if last_batch > 0:
            print(f"Riprendendo dal batch {last_batch + 1} (già processati {len(all_user_metric_results)} utenti)")
        else:
            all_user_metric_results = {}
            all_held_out_items = {}
        
        final_evaluation = None
        
        # Determina da quale batch iniziare
        start_batch_index = last_batch if last_batch > 0 else 0
        
        # Processamento a batch
        for batch_start in range(start_batch_index * batch_size, total_users, batch_size):
            batch_end = min(batch_start + batch_size, total_users)
            batch_user_ids = all_user_ids[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total_users + batch_size - 1) // batch_size
            
            print(f"\n--- BATCH {batch_num}/{total_batches}: utenti {batch_start+1}-{batch_end} ---")
            print(f"DEBUG: batch_user_ids = {batch_user_ids[:5]}...")  # Mostra i primi 5 user IDs
            
            # Stato iniziale per questo batch - DEVE avere tutti i campi di RecommenderState
            batch_state = {
                # Informazioni sull'utente corrente
                "user_id": None,
                "user_profile": None,
                
                # Cataloghi specifici per metrica
                "catalog_precision": None,
                "catalog_coverage": None,
                
                # Risultati delle metriche (separate keys per evitare concurrent updates)
                "precision_at_k_result": None,
                "coverage_result": None,
                "precision_completed": False,  # Flag per tracking completion
                "coverage_completed": False,   # Flag per tracking completion
                "metric_results": {},
                "metric_tasks_completed": 0,
                "expected_metrics": len(self.current_prompt_variants),
                
                # Gestione multi-utente
                "current_user_index": 0,  # Sempre 0 per ogni batch (gli user_ids sono già slice)
                "batch_user_ids": batch_user_ids,  # Solo gli utenti di questo batch (nuovo campo)
                "user_ids": all_user_ids,  # Lista completa per compatibilità
                "all_user_results": {},
                
                # Output finale
                "final_evaluation": None,
                
                # Dati per valutazione
                "held_out_items": {},
                
                # Gestione errori
                "error": None
            }
            
            # DEBUG: Verifica che batch_state abbia i dati corretti
            print(f"DEBUG batch_state creation: batch_user_ids length = {len(batch_state['batch_user_ids'])}")
            print(f"DEBUG batch_state creation: first 5 batch_user_ids = {batch_state['batch_user_ids'][:5]}")
            print(f"DEBUG batch_state creation: current_user_index = {batch_state['current_user_index']}")
            
            try:
                # Calcola limite di ricorsione dinamico basato sulla dimensione del batch
                # Molto più conservativo per evitare errori di ricorsione
                recursion_limit = max(50, len(batch_user_ids) * 3 + 30)
                config = {"recursion_limit": recursion_limit}
                
                print(f"Processando batch con {len(batch_user_ids)} utenti (limite ricorsione: {recursion_limit})")
                
                # Esegui il grafo LangGraph per questo batch
                batch_final_state = await self.recommender_graph.ainvoke(batch_state, config=config)
                
                # Raccogli risultati da questo batch
                batch_results = batch_final_state.get("all_user_results", {})
                batch_held_out = batch_final_state.get("held_out_items", {})
                batch_evaluation = batch_final_state.get("final_evaluation")
                
                # Unisci ai risultati complessivi
                all_user_metric_results.update(batch_results)
                all_held_out_items.update(batch_held_out)
                
                # Mantieni l'ultima valutazione finale (o puoi combinarle)
                if batch_evaluation:
                    final_evaluation = batch_evaluation
                
                print(f"Batch {batch_num} completato: {len(batch_results)} utenti processati")
                
                # Salva checkpoint dopo ogni batch per evitare perdite di progresso
                self._save_checkpoint(batch_num, all_user_metric_results, all_held_out_items)
                
            except Exception as e:
                print(f"Errore nel batch {batch_num} (utenti {batch_start+1}-{batch_end}): {e}")
                traceback.print_exc()
                # Continua con il prossimo batch invece di fermarsi
                continue
        
        # Se non abbiamo valutazione finale, creane una di emergenza
        if not final_evaluation:
            final_evaluation = {
                "final_recommendations": [], 
                "justification": "Processamento completato a batch - valutazione non disponibile", 
                "trade_offs": "N/A"
            }
        
        end_all_users = time.time()
        print(f"\nProcessamento completato!")
        print(f"Utenti totali processati: {len(all_user_metric_results)}/{total_users}")
        print(f"Tempo totale: {end_all_users - start_all_users:.2f} secondi")
        
        # Ripristina le varianti di prompt di default
        self.current_prompt_variants = PROMPT_VARIANTS.copy()
        
        return all_user_metric_results, final_evaluation, all_held_out_items

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

    def calculate_and_display_metrics(self, metric_results: Dict, final_evaluation: Dict, per_user_relevant_items: Dict[int, List[int]], experiment_name: Optional[str] = None) -> Dict:
        """Calcola e visualizza metriche per utente e aggregate.
        
        Args:
            metric_results: Dizionario {user_id: {metric_name: results}}
            final_evaluation: Dizionario con le raccomandazioni finali aggregate.
            per_user_relevant_items: Dizionario {user_id: list_of_held_out_ids}
            experiment_name: Nome dell'esperimento, se applicabile, per guidare calcoli di metriche specifiche.
            
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
        
        # current_prompt_variants potrebbe contenere i nomi delle strategie usate nell'esperimento corrente
        # metric_names sarà la lista di queste strategie (es. ['precision_at_k_recency', 'coverage_standard'])
        # o i nomi di default se non è un esperimento custom (es. ['precision_at_k', 'coverage'])
        metric_names_in_experiment = list(self.current_prompt_variants.keys()) if self.current_prompt_variants else \
                       list(PROMPT_VARIANTS.keys())


        per_user_calculated_metrics, aggregate_calculated_metrics = metrics_calculator.compute_all_metrics(
            metric_results_for_users=metric_results,
            per_user_relevant_items=per_user_relevant_items,
            k_values=k_values,
            metric_names=metric_names_in_experiment, # Passa i nomi delle metriche dell'esperimento corrente
            experiment_name=experiment_name, # Passa experiment_name
            final_evaluation_recommendations=final_evaluation.get('final_recommendations', []) # NUOVA RIGA
        )

        # --- Logica di Visualizzazione (utilizza i dati calcolati) ---
        print("\nMetriche Calcolate (Per Utente):")
        for user_id, u_calculated_data in per_user_calculated_metrics.items():
            if not per_user_relevant_items.get(user_id, []):
                print(f"  Utente {user_id}: Attenzione - Nessun item rilevante (held-out) di riferimento.")
            
            print(f"  Utente {user_id}:")
            # Itera sulle metric_names_in_experiment per coerenza con compute_all_metrics
            for metric_name_key in metric_names_in_experiment: 
                data_for_metric = u_calculated_data.get(metric_name_key)
                if data_for_metric:
                    pak_scores = data_for_metric.get("precision_scores", {}) # Default a {}
                    genre_cov = data_for_metric.get("genre_coverage", 0.0) # Default a 0.0
                    pak_str = ", ".join([f"P@{k}={score:.4f}" for k, score in pak_scores.items()])
                    
                    # Visualizza le nuove metriche specifiche se presenti
                    avg_year_str = ""
                    if "average_release_year" in data_for_metric:
                        avg_year_str = f", AvgYear={data_for_metric['average_release_year']:.1f}"
                    
                    temp_disp_str = ""
                    if "temporal_dispersion" in data_for_metric:
                        temp_disp_str = f", TempDisp={data_for_metric['temporal_dispersion']:.2f}"
                        
                    genre_entropy_str = ""
                    if "genre_entropy" in data_for_metric:
                        genre_entropy_str = f", GenreEntropy={data_for_metric['genre_entropy']:.4f}"

                    print(f"    {metric_name_key}: {pak_str}, GenreCoverage={genre_cov:.4f}{avg_year_str}{temp_disp_str}{genre_entropy_str}")
                else:
                    print(f"    {metric_name_key}: Dati non disponibili.")

        print("\nMetriche Aggregate (Medie su Utenti):")
        # Itera su metric_names_in_experiment + ['final'] per mantenere l'ordine e includere le metriche finali
        for name_key in metric_names_in_experiment + ['final']:
            agg_data = aggregate_calculated_metrics.get(name_key)
            if agg_data:
                label = f"Mean {name_key.capitalize()}" if name_key != 'final' else "Final Aggregated"
                
                # Visualizzazione P@k/MAP@k e Genre Coverage esistente
                if name_key == 'final':
                    pak_scores_agg = agg_data.get("precision_scores_agg", {})
                    genre_cov_agg = agg_data.get("genre_coverage", 0.0)
                    pak_str_agg = ", ".join([f"P@{k}={score:.4f}" for k, score in pak_scores_agg.items()])
                    print(f"  {label}: {pak_str_agg} (vs all held-out), GenreCoverage={genre_cov_agg:.4f}")
                else: # Per le strategie di prompt
                    map_scores = agg_data.get("map_at_k", {})
                    mean_genre_cov = agg_data.get("mean_genre_coverage", 0.0)
                    map_str = ", ".join([f"MAP@{k}={score:.4f}" for k, score in map_scores.items()])
                      # Visualizza le nuove metriche aggregate se presenti
                    avg_year_agg_str = ""
                    if "average_release_year" in agg_data: # Nome chiave per aggregato
                        avg_year_agg_str = f", AvgYear={agg_data['average_release_year']:.1f}"
                    elif "avg_year" in agg_data: # Nome chiave alternativo per agente aggregatore
                        avg_year_agg_str = f", AvgYear={agg_data['avg_year']:.1f}"
                    
                    temp_disp_agg_str = ""
                    if "temporal_dispersion" in agg_data: # Nome chiave per aggregato
                        temp_disp_agg_str = f", TempDisp={agg_data['temporal_dispersion']:.2f}"
                    elif "avg_temporal_dispersion" in agg_data: # Nome chiave alternativo per agente aggregatore
                        temp_disp_agg_str = f", TempDisp={agg_data['avg_temporal_dispersion']:.2f}"
                        
                    genre_entropy_agg_str = ""
                    if "genre_entropy" in agg_data: # Nome chiave per aggregato
                        genre_entropy_agg_str = f", GenreEntropy={agg_data['genre_entropy']:.4f}"
                    elif "avg_genre_entropy" in agg_data: # Nome chiave alternativo per agente aggregatore
                        genre_entropy_agg_str = f", GenreEntropy={agg_data['avg_genre_entropy']:.4f}"
                        
                    print(f"  {label}: {map_str}, Mean GenreCoverage={mean_genre_cov:.4f}{avg_year_agg_str}{temp_disp_agg_str}{genre_entropy_agg_str}")
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
        
    async def generate_standard_recommendations(self, batch_size: int = 50) -> Dict:
        """Genera raccomandazioni standard, calcola metriche e salva.
        
        Args:
            batch_size: Numero di utenti da processare per batch (default: 50)
        """
        if not self.experiment_manager:
            self.experiment_manager = ExperimentManager(self)
        return await self.experiment_manager.run_standard_pipeline(batch_size=batch_size)

    def _save_checkpoint(self, batch_num: int, all_user_results: Dict, all_held_out: Dict) -> None:
        """Salva un checkpoint del progresso per recupero in caso di errori."""
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "batch_num": batch_num,
                "completed_users": len(all_user_results),
                "user_results": {str(k): v for k, v in all_user_results.items()},
                "held_out_items": {str(k): v for k, v in all_held_out.items()}
            }
            
            checkpoint_path = f"checkpoint_batch_{batch_num}.json"
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            print(f"Checkpoint salvato: {checkpoint_path}")
        except Exception as e:
            print(f"Errore nel salvataggio checkpoint: {e}")

    def _load_latest_checkpoint(self) -> Tuple[int, Dict, Dict]:
        """Carica l'ultimo checkpoint disponibile."""
        try:
            import glob
            checkpoint_files = glob.glob("checkpoint_batch_*.json")
            if not checkpoint_files:
                return 0, {}, {}
            
            # Trova il checkpoint più recente
            latest_file = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            
            with open(latest_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            
            user_results = {int(k): v for k, v in checkpoint_data["user_results"].items()}
            held_out = {int(k): v for k, v in checkpoint_data["held_out_items"].items()}
            
            print(f"Checkpoint caricato: {latest_file}, batch {checkpoint_data['batch_num']}, {checkpoint_data['completed_users']} utenti completati")
            return checkpoint_data["batch_num"], user_results, held_out
            
        except Exception as e:
            print(f"Errore nel caricamento checkpoint: {e}")
            return 0, {}, {}

    def _clean_llm_json_response(self, response_content: str) -> str:
        """Pulisce la risposta dell'LLM rimuovendo markdown e altri elementi che causano errori di parsing JSON."""
        if not isinstance(response_content, str):
            return str(response_content)
        
        # Rimuovi markdown code blocks
        import re
        # Pattern più robusto per rimuovere ```json ... ``` or ``` ... ```
        cleaned = re.sub(r'```(?:json|JSON)?\s*\n?(.*?)\n?```', r'\1', response_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Rimuovi commenti JavaScript-style (// commenti)
        cleaned = re.sub(r'//.*$', '', cleaned, flags=re.MULTILINE)
        
        # Rimuovi spazi all'inizio e alla fine
        cleaned = cleaned.strip()
        
        # Se ancora non inizia con {, prova a trovare il primo { e l'ultimo }
        if not cleaned.startswith('{'):
            start_idx = cleaned.find('{')
            if start_idx != -1:
                end_idx = cleaned.rfind('}')
                if end_idx != -1 and end_idx > start_idx:
                    cleaned = cleaned[start_idx:end_idx+1]
        
        # Rimuovi virgole finali prima delle parentesi graffe di chiusura
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        # Rimuovi caratteri di controllo non stampabili
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
        
        return cleaned