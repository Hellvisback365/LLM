import json
import os
from src.recommender.core.metrics_calculator import calculate_metrics_for_recommendations, add_metrics_to_results

def test_metrics_calculation():
    print("=== Test del calcolo delle metriche ===\n")
    
    # Controlla se esiste il file dei risultati
    recommendations_file = "recommendation_results.json"
    if not os.path.exists(recommendations_file):
        print(f"File {recommendations_file} non trovato. Esegui prima python agent.py")
        return False
    
    # Carica i risultati
    try:
        with open(recommendations_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        print("Risultati caricati correttamente.")
        
        # Estrai i dati necessari
        metric_recommendations = results.get("metric_recommendations", {})
        final_evaluation = results.get("final_evaluation", {})
        
        # Calcola le metriche
        print("\nCalcolo delle metriche in corso...")
        metrics = calculate_metrics_for_recommendations(metric_recommendations, final_evaluation)
        
        # Mostra un riepilogo
        print("\n=== Riepilogo delle metriche calcolate ===")
        
        print("\nPrecision@k:")
        print(f"  Precision_at_k: {metrics.get('precision_at_k', {}).get('precision_score', 0):.4f}")
        print(f"  Coverage: {metrics.get('coverage', {}).get('precision_score', 0):.4f}")
        print(f"  Final: {metrics.get('final_recommendations', {}).get('precision_score', 0):.4f}")
        
        print("\nGenre Coverage:")
        print(f"  Precision_at_k: {metrics.get('precision_at_k', {}).get('genre_coverage', 0):.4f}")
        print(f"  Coverage: {metrics.get('coverage', {}).get('genre_coverage', 0):.4f}")
        print(f"  Final: {metrics.get('final_recommendations', {}).get('genre_coverage', 0):.4f}")
        
        print(f"\nCoverage totale: {metrics.get('total_coverage', 0):.4f}")
        
        # Aggiungi le metriche ai risultati
        print("\nAggiunta delle metriche al file dei risultati...")
        success = add_metrics_to_results(metrics)
        
        if success:
            print("Metriche aggiunte con successo al file dei risultati.")
        else:
            print("Errore nell'aggiunta delle metriche al file dei risultati.")
        
        return True
    
    except Exception as e:
        print(f"Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_metrics_calculation() 