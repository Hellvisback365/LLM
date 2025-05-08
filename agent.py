# ----------------------------
# Import e setup iniziale
# ----------------------------
import os
import asyncio
from typing import Dict
from dotenv import load_dotenv

# ----------------------------
# Import dei moduli locali
# ----------------------------
# Importa il sistema di raccomandazione unificato
from src.recommender.core.recommender import RecommenderSystem 
# Modifica l'importazione per prelevare PROMPT_VARIANTS dal nuovo modulo prompt_manager
from src.recommender.core.prompt_manager import PROMPT_VARIANTS
from src.reporting.experiment_reporter import ExperimentReporter


# ----------------------------
# Setup ambiente (solo load_dotenv)
# ----------------------------
load_dotenv()

# ----------------------------
# 1. Main (Modificato per usare RecommenderSystem dal core)
# ----------------------------
async def main():
    print("\n=== Starting Unified Recommender System (via agent.py) ===\n")
    recommender = RecommenderSystem(specific_user_ids=[4277, 4169, 1680])
    try:
         recommender.initialize_system(force_reload_data=False, force_recreate_vector_store=False)
    except Exception as e:
         print(f"Errore fatale inizializzazione: {e}. Impossibile continuare."); return
         
    try:
        result = await recommender.generate_standard_recommendations()
        print("\n=== Standard Recommendation Process Complete ===")
        if result and isinstance(result, dict) and result.get('final_evaluation'):
             print(f"Final recommendations: {result['final_evaluation'].get('final_recommendations', 'N/A')}")
        else:
             print("Processo completato ma senza risultati finali validi.")
             print(f"Output ricevuto: {result}")
    except Exception as e:
        print(f"Errore pipeline standard: {e}")
        import traceback; traceback.print_exc()

    run_experiments_flag = os.getenv("RUN_EXPERIMENTS", "false").lower() == "true"
    if run_experiments_flag:
        print("\n=== Running Prompt Variant Experiments ===\n")
        precision_variants = {
            "precision_at_k_serendipity": (
                "You are an expert recommendation system that optimizes for PRECISION@K with focus on SERENDIPITY..."
            ),
            "precision_at_k_recency": (
                "You are an expert recommendation system that optimizes for PRECISION@K with focus on RECENCY..."
            )
        }
        coverage_variants = {
            "coverage_genre_balance": (
                "You are an expert recommendation system that optimizes for COVERAGE with GENRE BALANCE..."
            ),
            "coverage_temporal": (
                "You are an expert recommendation system that optimizes for TEMPORAL COVERAGE..."
            )
        }
        
        experiment_tasks = []
        # Prepara task esperimenti
        for name, prompt in precision_variants.items():
            variant_dict = {"precision_at_k": prompt, "coverage": PROMPT_VARIANTS["coverage"]}
            experiment_tasks.append(recommender.generate_recommendations_with_custom_prompt(variant_dict, name))
        for name, prompt in coverage_variants.items():
            variant_dict = {"coverage": prompt, "precision_at_k": PROMPT_VARIANTS["precision_at_k"]}
            experiment_tasks.append(recommender.generate_recommendations_with_custom_prompt(variant_dict, name))
        combined = {
            "precision_at_k": precision_variants["precision_at_k_serendipity"],
            "coverage": coverage_variants["coverage_temporal"]
        }
        experiment_tasks.append(recommender.generate_recommendations_with_custom_prompt(combined, "combined_serendipity_temporal"))
        
        try:
             print(f"Avvio di {len(experiment_tasks)} esperimenti...")
             experiment_results = await asyncio.gather(*experiment_tasks, return_exceptions=True)
             print("\n--- Esperimenti Eseguiti ---")
             for i, res_tuple in enumerate(experiment_results):
                  if isinstance(res_tuple, Exception):
                       print(f"Errore nell'esperimento {i}: {res_tuple}")
                  elif isinstance(res_tuple, tuple) and len(res_tuple) == 2:
                       print(f"  -> Esperimento salvato in: {res_tuple[1]}")
                  else:
                       print(f"Risultato inatteso esperimento {i}: {res_tuple}")
        except Exception as e:
             print(f"Errore esecuzione esperimenti: {e}")

        try:
            print("\n=== Generating Experiment Reports ===")
            reporter = ExperimentReporter(experiments_dir="experiments")
            if reporter.experiments: # Controlla se ci sono esperimenti da analizzare
                summary = reporter.run_comprehensive_analysis(output_dir="reports")
                print("\nAnalysis complete. Reports generated:")
                for report_path in summary.get("reports_generated_status", {}).values():
                    if isinstance(report_path, str): print(f"- {report_path}")
                print(f"\nTotal experiments analyzed: {summary.get('total_experiments', 0)}")
            else:
                print("Nessun file di esperimento trovato o caricato correttamente per generare i report.")
        except Exception as e:
             print(f"Errore generazione report: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nEsecuzione interrotta.")
    except Exception as e:
        print(f"\nErrore non gestito: {e}")
        import traceback; traceback.print_exc()