"""
Nodi del grafo LangGraph per il sistema di raccomandazione.
Questo modulo contiene l'implementazione dei nodi utilizzati nel grafo LangGraph
per orchestrare il processo di raccomandazione.
"""

import json
import time
from typing import Dict, List, Any, TypedDict, Optional

# Definizione dello stato utilizzato nel grafo
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

class RecommenderGraphNodes:
    """Implementazione dei nodi per il grafo LangGraph del sistema di raccomandazione."""
    
    def __init__(self, recommender):
        """
        Inizializza i nodi del grafo con riferimento al RecommenderSystem principale.
        
        Args:
            recommender: Istanza di RecommenderSystem 
        """
        self.recommender = recommender
    
    async def node_initialize(self, state: RecommenderState) -> Dict[str, Any]:
        """Inizializza lo stato del workflow."""
        if not self.recommender.datasets_loaded:
            self.recommender._load_datasets()
        
        return {
            "user_ids": self.recommender.specific_user_ids,
            "current_user_index": 0,
            "all_user_results": {},
            "metric_results": {},
            "precision_at_k_result": None,
            "coverage_result": None,
            "precision_completed": False,
            "coverage_completed": False,
            "metric_tasks_completed": 0,
            "expected_metrics": len(self.recommender.current_prompt_variants),
            "held_out_items": {},
            "final_evaluation": None,
            "error": None
        }

    async def node_prepare_user_data(self, state: RecommenderState) -> Dict[str, Any]:
        """Prepara i dati dell'utente corrente."""
        user_index = state["current_user_index"]
        user_ids = state["user_ids"]
        
        if user_index >= len(user_ids):
            return {"error": "Indice utente fuori range"}
        
        user_id = user_ids[user_index]
        
        # Recupera il profilo utente
        if user_id not in self.recommender.user_profiles.index:
            return {"error": f"Utente {user_id} non trovato."}
            
        profile_series = self.recommender.user_profiles.loc[user_id]
        
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
        catalog_json_fallback = self.recommender.get_optimized_catalog(limit=300)
        
        if self.recommender.rag:
            try:
                # Catalogo per Precision@k
                print(f"RAG: Tentativo chiamata similarity_search per precision_at_k per utente {user_id}...")
                start_rag_p = time.time()
                cat_p = self.recommender.rag.similarity_search(profile_summary, k=300, metric_focus="precision_at_k", user_id=int(user_id))
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
                cat_c = self.recommender.rag.similarity_search(coverage_query, k=300, metric_focus="coverage", user_id=int(user_id))
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

    async def node_run_precision_metric(self, state: RecommenderState) -> Dict[str, Any]:
        """Esegue la logica per la metrica precision_at_k."""
        user_id_for_log = state.get('user_id') # Prendi user_id dallo stato
        print(f"Executing precision_at_k metric for user {user_id_for_log}..." )
        
        metric_name = "precision_at_k"
        catalog = state["catalog_precision"]
        user_profile = state["user_profile"]
        
        # Riutilizza la stessa logica del tool esistente, passando user_id
        result = await self.recommender._run_metric_tool_internal(metric_name, catalog, user_profile, user_id=user_id_for_log)
        
        # Usa precision_at_k_result invece di metric_results
        return {
            "precision_at_k_result": result,
            "precision_completed": True
        }

    async def node_run_coverage_metric(self, state: RecommenderState) -> Dict[str, Any]:
        """Esegue la logica per la metrica coverage."""
        user_id_for_log = state.get('user_id') # Prendi user_id dallo stato
        print(f"Executing coverage metric for user {user_id_for_log}...")
        
        metric_name = "coverage"
        catalog = state["catalog_coverage"]
        user_profile = state["user_profile"]
        
        # Riutilizza la stessa logica del tool esistente, passando user_id
        result = await self.recommender._run_metric_tool_internal(metric_name, catalog, user_profile, user_id=user_id_for_log)
        
        # Usa coverage_result invece di metric_results
        return {
            "coverage_result": result,
            "coverage_completed": True
        }

    async def node_collect_metric_results(self, state: RecommenderState) -> Dict[str, Any]:
        """Raccoglie i risultati delle metriche per l'utente corrente."""
        user_id = state["user_id"]
        
        # Checkpoint: verifica se entrambe le metriche sono completate
        if not (state.get("precision_completed", False) and state.get("coverage_completed", False)):
            # Aspetta che entrambe siano completate
            return state
        
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
        
        # Incrementa l'indice utente e resetta i flag per il prossimo utente
        return {
            "all_user_results": all_results,
            "metric_tasks_completed": completed_count,
            "current_user_index": state["current_user_index"] + 1,
            # Reset dei risultati delle singole metriche per il prossimo utente
            "precision_at_k_result": None,
            "coverage_result": None,
            "precision_completed": False,
            "coverage_completed": False
        }

    async def node_next_user_or_finish(self, state: RecommenderState) -> Dict[str, Any]:
        """Determina se passare all'utente successivo o terminare."""
        # Questo nodo non è più necessario con i conditional_edges ma lo manteniamo per chiarezza
        return {}

    async def node_evaluate_all_results(self, state: RecommenderState) -> Dict[str, Any]:
        """Valuta i risultati di tutti gli utenti."""
        all_results = state["all_user_results"]
        
        # Converti in formato JSON per l'evaluator
        all_results_str = json.dumps(all_results, ensure_ascii=False, indent=2)
        eval_catalog = self.recommender.get_optimized_catalog(limit=300)
        
        print("\n--- Valutazione Aggregata ---")
        start_eval = time.time()
        
        # Chiama la stessa logica dell'evaluator esistente
        final_evaluation = await self.recommender._evaluate_recommendations_internal(
            all_recommendations_str=all_results_str,
            catalog_str=eval_catalog
        )
        
        end_eval = time.time()
        print(f"Tempo impiegato per valutazione aggregata: {end_eval - start_eval:.2f} secondi")
        print(f"Final recommendations: {final_evaluation.get('final_recommendations', [])}")
        
        return {
            "final_evaluation": final_evaluation
        }
    
    def check_metrics_completion(self, state: RecommenderState) -> str:
        """Verifica se tutte le metriche sono state calcolate."""
        # Usa i flag di completamento invece del contatore
        all_completed = state.get("precision_completed", False) and state.get("coverage_completed", False)
        
        if all_completed:
            return "complete"
        return "not_complete"

    def check_users_completion(self, state: RecommenderState) -> str:
        """Verifica se tutti gli utenti sono stati elaborati."""
        if state["current_user_index"] >= len(state["user_ids"]):
            return "evaluate"
        return "next_user"

# Importazione alla fine per evitare dipendenze circolari
import pandas as pd 