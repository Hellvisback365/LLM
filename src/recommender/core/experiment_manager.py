"""
Gestore degli esperimenti di raccomandazione.

Questo modulo fornisce una classe che centralizza la gestione degli esperimenti
con diverse varianti di prompt e la raccolta/salvataggio dei risultati.
"""

import os
import json
import asyncio
import sys
from datetime import datetime
from typing import Dict, Tuple

from src.recommender.utils.data_utils import convert_numpy_types_for_json

class ExperimentManager:
    """
    Gestisce l'esecuzione di esperimenti di raccomandazione e il salvataggio dei risultati.
    
    Questa classe è una facade che semplifica l'interazione con il sistema di raccomandazione
    per eseguire esperimenti con diverse configurazioni di prompt.
    """
    
    def __init__(self, recommender_system):
        """
        Inizializza il gestore degli esperimenti.
        
        Args:
            recommender_system: Istanza del sistema di raccomandazione
        """
        self.recommender = recommender_system
        
    async def run_experiment(self, prompt_variants: Dict, experiment_name: str = "custom_experiment") -> Tuple[Dict, str]:
        """
        Esegue un esperimento con prompt customizzati.
        
        Args:
            prompt_variants: Dizionario con le varianti di prompt da utilizzare
            experiment_name: Nome dell'esperimento per il salvataggio dei risultati
            
        Returns:
            Tuple[Dict, str]: Risultati dell'esperimento e percorso del file salvato
        """
        print(f"\n=== Esecuzione Esperimento: {experiment_name} ===")
        
        # Esegui la pipeline con le varianti custom
        metric_results, final_evaluation, per_user_held_out_items = await self.recommender.run_recommendation_pipeline(
            use_prompt_variants=prompt_variants
        )
        
        # Calcola metriche
        metrics = self.recommender.calculate_and_display_metrics(
            metric_results, final_evaluation, per_user_held_out_items
        )
        
        # Salva risultati
        os.makedirs("experiments", exist_ok=True)
        filename = f"experiments/experiment_{experiment_name}.json"
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "experiment_info": {"name": experiment_name, "prompt_variants": prompt_variants},
            "metric_recommendations": metric_results,
            "final_evaluation": final_evaluation,
            "metrics": metrics,
            "per_user_held_out_items": {str(k): v for k, v in per_user_held_out_items.items()}
        }
        
        # Converti np.float64 in float nativo per JSON
        result_to_save = convert_numpy_types_for_json(result)
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(result_to_save, f, ensure_ascii=False, indent=2)
            print(f"Risultati esperimento salvati: {filename}")
        except Exception as e:
            print(f"Errore salvataggio file esperimento {filename}: {e}")
            
        return result, filename
        
    async def run_standard_pipeline(self) -> Dict:
        """
        Esegue la pipeline standard di raccomandazione.
        
        Returns:
            Dict: Risultati della pipeline standard
        """
        print("\n=== Esecuzione Pipeline Standard ===")
        
        # Esegui la pipeline con le varianti standard
        metric_results, final_evaluation, per_user_held_out_items = await self.recommender.run_recommendation_pipeline()
        
        # Calcola metriche
        metrics = self.recommender.calculate_and_display_metrics(
            metric_results, final_evaluation, per_user_held_out_items
        )
        
        # Salva risultati
        self.recommender.save_results(
            metric_results, final_evaluation, 
            metrics_calculated=metrics, 
            per_user_held_out_items=per_user_held_out_items
        )
        
        # Dà tempo all'event loop di stabilizzarsi prima di stampare i messaggi finali
        await asyncio.sleep(0.1)
        sys.stdout.flush()
        
        # Stampa risultati finali
        print("\n=== Standard Recommendation Process Complete ===")
        print(f"Final recommendations: {final_evaluation.get('final_recommendations', [])}")
        sys.stdout.flush()
        
        # Converti np.float64 in float nativo prima di restituire
        return convert_numpy_types_for_json({
            "timestamp": datetime.now().isoformat(),
            "metric_recommendations": metric_results,
            "final_evaluation": final_evaluation,
            "metrics": metrics
        }) 