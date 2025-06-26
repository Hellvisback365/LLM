"""
Nodi del grafo LangGraph per il sistema di raccomandazione.
Questo modulo contiene l'implementazione dei nodi utilizzati nel grafo LangGraph
per orchestrare il processo di raccomandazione.
"""

import json
import time
import traceback
from typing import Dict, List, Any, TypedDict, Optional, Annotated
from langgraph.graph import add_messages

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
    
    # Gestione multi-utente - usiamo batch_user_ids invece di user_ids globali
    current_user_index: int
    batch_user_ids: Optional[List[int]]  # Lista specifica del batch corrente
    user_ids: List[int]  # Lista completa (per compatibilità)
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

        # FIX: Gestisce il caso in cui non vengono forniti ID utente specifici
        user_ids = self.recommender.specific_user_ids
        if user_ids is None:
            # Se non ci sono ID specifici, usa tutti gli utenti dai profili caricati
            if self.recommender.user_profiles is not None and not self.recommender.user_profiles.empty:
                user_ids = self.recommender.user_profiles.index.tolist()
            else:
                # Fallback a una lista vuota se i profili non sono ancora stati caricati
                user_ids = []
        
        return {
            "user_ids": user_ids,
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
        
        # Usa batch_user_ids se disponibile, altrimenti fallback a user_ids
        batch_user_ids = state.get("batch_user_ids")
        if batch_user_ids is not None:
            user_ids = batch_user_ids
            print(f"DEBUG node_prepare_user_data: USING BATCH_USER_IDS, user_index={user_index}, batch_length={len(user_ids)}")
        else:
            user_ids = state["user_ids"]
            print(f"DEBUG node_prepare_user_data: USING GLOBAL USER_IDS, user_index={user_index}, user_ids_length={len(user_ids)}")
        
        print(f"DEBUG node_prepare_user_data: first 5 user_ids = {user_ids[:5] if len(user_ids) >= 5 else user_ids}")
        
        if user_index >= len(user_ids):
            print("ERROR: user_index >= len(user_ids) - this should not happen!")
            return {"error": "Indice utente fuori range"}
        
        user_id = user_ids[user_index]
        print(f"DEBUG node_prepare_user_data: processing user_id={user_id} (at index {user_index})")
        
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
        catalog_json_fallback = self.recommender.get_optimized_catalog(limit=25)
        
        if self.recommender.rag:
            try:
                # MODIFICA: Query più specifica per Precision@k
                precision_query = f"Film that the user will definitely love—high probability of a positive rating based on preferences {profile_summary}"
                print(f"RAG PRE-CALL: Tentativo chiamata similarity_search per precision_at_k per utente {user_id}...")
                start_rag_p = time.time()
                cat_p = self.recommender.rag.similarity_search(precision_query, k=50, metric_focus="precision_at_k", user_id=int(user_id))
                end_rag_p = time.time()
                print(f"RAG POST-CALL: Tempo impiegato per precision_at_k: {end_rag_p - start_rag_p:.2f} secondi. Risultati: {len(cat_p) if cat_p is not None else 'None'}")
                catalog_precision = json.dumps(cat_p[:25], ensure_ascii=False) if cat_p else catalog_json_fallback
                print(f"RAG: Generato catalogo specifico per precision_at_k (size: {len(cat_p[:25]) if cat_p else 'Fallback'}) (Successo)")
            except Exception as e:
                print(f"!!! ERRORE RAG (precision_at_k) user {user_id}: {type(e).__name__} - {e} !!!")
                traceback.print_exc()
                catalog_precision = catalog_json_fallback

            try:
                # MODIFICA: Query completamente diversa per Coverage - senza profilo utente per massimizzare diversità
                coverage_query = "Diverse films — maximum variety, complete catalog, exploration of a wide range of different types"
                print(f"RAG PRE-CALL: Tentativo chiamata similarity_search per coverage per utente {user_id}...")
                start_rag_c = time.time()
                # MODIFICA: Non passiamo user_id per coverage per evitare bias verso preferenze utente
                cat_c = self.recommender.rag.similarity_search(coverage_query, k=50, metric_focus="coverage", user_id=None)
                end_rag_c = time.time()
                print(f"RAG POST-CALL: Tempo impiegato per coverage: {end_rag_c - start_rag_c:.2f} secondi. Risultati: {len(cat_c) if cat_c is not None else 'None'}")
                catalog_coverage = json.dumps(cat_c[:25], ensure_ascii=False) if cat_c else catalog_json_fallback
                print(f"RAG: Generato catalogo specifico per coverage (size: {len(cat_c[:25]) if cat_c else 'Fallback'}) (Successo)")
            except Exception as e:
                print(f"!!! ERRORE RAG (coverage) user {user_id}: {type(e).__name__} - {e} !!!")
                traceback.print_exc()
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
        
        # Con l'esecuzione sequenziale, entrambe le metriche dovrebbero essere completate
        precision_result = state.get("precision_at_k_result", {})
        coverage_result = state.get("coverage_result", {})
        
        combined_results = {}
        if precision_result:
            combined_results["precision_at_k"] = precision_result
        if coverage_result:
            combined_results["coverage"] = coverage_result
        
        print(f"Raccolti risultati metriche per utente {user_id}: {', '.join(combined_results.keys())}")
        
        # Aggiorna i risultati complessivi per utente
        all_results = state["all_user_results"].copy()
        all_results[user_id] = combined_results
        
        # Incrementa l'indice utente e resetta i dati per il prossimo utente
        new_index = state["current_user_index"] + 1
        
        print(f"DEBUG collect_metric_results: incrementing user index from {state['current_user_index']} to {new_index}")
        print(f"DEBUG collect_metric_results: total users = {len(state['user_ids'])}")
        
        return {
            "all_user_results": all_results,
            "current_user_index": new_index,
            # Reset per il prossimo utente
            "precision_at_k_result": None,
            "coverage_result": None,
            "precision_completed": False,
            "coverage_completed": False,
            "user_id": None,
            "user_profile": None,
            "catalog_precision": None,
            "catalog_coverage": None
        }

    async def node_next_user_or_finish(self, state: RecommenderState) -> Dict[str, Any]:
        """Determina se passare all'utente successivo o terminare."""
        # Questo nodo non è più necessario con i conditional_edges ma lo manteniamo per chiarezza
        return {}

    async def node_evaluate_all_results(self, state: RecommenderState) -> Dict[str, Any]:
        """Valuta i risultati di tutti gli utenti."""
        all_results = state["all_user_results"]
        
        # SOLUZIONE 2: Ristruttura per metrica invece che per utente
        restructured_data = {
            "precision_at_k_recommendations": {},
            "coverage_recommendations": {}
        }
        
        # Aggrega tutte le raccomandazioni per metrica
        for user_id, user_results in all_results.items():
            if "precision_at_k" in user_results:
                restructured_data["precision_at_k_recommendations"][f"user_{user_id}"] = {
                    "recommendations": user_results["precision_at_k"]["recommendations"],
                    "explanation": user_results["precision_at_k"]["explanation"]
                }
            if "coverage" in user_results:
                restructured_data["coverage_recommendations"][f"user_{user_id}"] = {
                    "recommendations": user_results["coverage"]["recommendations"], 
                    "explanation": user_results["coverage"]["explanation"]
                }
        
        # Converti in formato JSON per l'evaluator
        all_results_str = json.dumps(restructured_data, ensure_ascii=False, indent=2)
        eval_catalog = self.recommender.get_optimized_catalog(limit=100)
        
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
        current_index = state["current_user_index"]
        
        # Determina se stiamo usando batch processing
        if "batch_user_ids" in state and state["batch_user_ids"]:
            total_users = len(state["batch_user_ids"])
            print(f"DEBUG check_users_completion (BATCH): current_index={current_index}, batch_size={total_users}")
        else:
            total_users = len(state["user_ids"])
            print(f"DEBUG check_users_completion (FULL): current_index={current_index}, total_users={total_users}")
        
        if current_index >= total_users:
            print("→ All users completed, proceeding to evaluation")
            return "evaluate"
        else:
            print(f"→ Processing next user (index {current_index})")
            return "next_user"
    
    async def node_synchronize_metrics(self, state: RecommenderState) -> Dict[str, Any]:
        """Nodo di sincronizzazione che aspetta il completamento di entrambe le metriche."""
        # Questo nodo verifica che entrambe le metriche siano completate
        precision_completed = state.get("precision_completed", False)
        coverage_completed = state.get("coverage_completed", False)
        
        print(f"Synchronization check: precision={precision_completed}, coverage={coverage_completed}")
        
        if precision_completed and coverage_completed:
            # Entrambe le metriche sono completate, può procedere
            print("Both metrics completed, proceeding to collect results")
            return {}
        else:
            # Non tutte le metriche sono completate, aspetta
            print("Waiting for metrics completion...")
            return {}

# Importazione alla fine per evitare dipendenze circolari
import pandas as pd