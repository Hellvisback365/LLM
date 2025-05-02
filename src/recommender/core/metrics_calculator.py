import os
import json
import pandas as pd
from typing import Dict, List, Any, Set
from src.recommender.utils.rag_utils import calculate_precision_at_k, calculate_coverage

def calculate_metrics_for_recommendations(metric_results: Dict, final_evaluation: Dict, experiment_file: str = None) -> Dict:
    """
    Calcola metriche quantitative per le raccomandazioni e le salva nel file dell'esperimento
    
    Args:
        metric_results: Risultati intermedi delle raccomandazioni per ogni metrica
        final_evaluation: Valutazione finale
        experiment_file: Percorso del file dell'esperimento dove salvare le metriche (opzionale)
        
    Returns:
        Dizionario con le metriche calcolate
    """
    try:
        print("\n=== Calcolo delle metriche quantitative ===")
        
        # Estrai le raccomandazioni
        precision_at_k_recs = metric_results.get('precision_at_k', {}).get('recommendations', [])
        coverage_recs = metric_results.get('coverage', {}).get('recommendations', [])
        final_recs = final_evaluation.get('final_recommendations', [])
        
        print(f"Raccomandazioni precision_at_k: {precision_at_k_recs}")
        print(f"Raccomandazioni coverage: {coverage_recs}")
        print(f"Raccomandazioni finali: {final_recs}")
        
        # Carica i dati dei film direttamente dal file JSON
        try:
            movies_path = os.path.join('data', 'processed', 'optimized_catalog.json')
            with open(movies_path, 'r', encoding='utf-8') as f:
                movies_data = json.load(f)
            
            # Converti in DataFrame
            movies = pd.DataFrame(movies_data)
            
            # Prepara i dati per il calcolo delle metriche
            all_movie_ids = movies['movie_id'].tolist() if 'movie_id' in movies.columns else []
            
            # Carica le valutazioni reali degli utenti invece di usare un proxy
            ratings_path = os.path.join('data', 'processed', 'filtered_ratings_specific.csv')
            relevant_items = []
            
            if os.path.exists(ratings_path):
                ratings_df = pd.read_csv(ratings_path)
                # Consideriamo rilevanti i film con valutazione >= 4
                relevant_items = ratings_df[ratings_df['rating'] >= 4]['movie_id'].unique().tolist()
                print(f"Trovati {len(relevant_items)} film rilevanti basati su valutazioni reali")
            else:
                print("File di valutazioni non trovato, utilizzo una simulazione come fallback")
                # Fallback al metodo precedente solo se il file non esiste
                relevant_items = all_movie_ids[:100] if len(all_movie_ids) >= 100 else all_movie_ids
            
            # Calcola precision@k
            precision_pak_value = calculate_precision_at_k(precision_at_k_recs, relevant_items)
            coverage_pak_value = calculate_precision_at_k(coverage_recs, relevant_items)
            final_pak_value = calculate_precision_at_k(final_recs, relevant_items)
            
            print(f"\nPrecision@k per precision_at_k: {precision_pak_value:.4f}")
            print(f"Precision@k per coverage: {coverage_pak_value:.4f}")
            print(f"Precision@k per raccomandazioni finali: {final_pak_value:.4f}")
            
            # Calcola coverage per ogni set di raccomandazioni
            precision_coverage = calculate_coverage(precision_at_k_recs, movies)
            coverage_metrics = calculate_coverage(coverage_recs, movies)
            final_coverage = calculate_coverage(final_recs, movies)
            
            print("\nCoverage per ogni set di raccomandazioni:")
            print(f"  precision_at_k - totale: {precision_coverage['total_coverage']:.4f}, generi: {precision_coverage['genre_coverage']:.4f}")
            print(f"  coverage - totale: {coverage_metrics['total_coverage']:.4f}, generi: {coverage_metrics['genre_coverage']:.4f}")
            print(f"  final - totale: {final_coverage['total_coverage']:.4f}, generi: {final_coverage['genre_coverage']:.4f}")
            
            # Calcola la coverage totale considerando tutte le raccomandazioni
            all_recs = list(set(precision_at_k_recs + coverage_recs + final_recs))
            total_metrics = calculate_coverage(all_recs, movies)
            
            print(f"\nCoverage totale del sistema:")
            print(f"  Totale film: {total_metrics['total_coverage']:.4f}")
            print(f"  Generi: {total_metrics['genre_coverage']:.4f}")
            
            # Crea il dizionario con tutte le metriche
            metrics = {
                "precision_at_k": {
                    "precision_score": precision_pak_value,
                    "total_coverage": precision_coverage['total_coverage'],
                    "genre_coverage": precision_coverage['genre_coverage']
                },
                "coverage": {
                    "precision_score": coverage_pak_value,
                    "total_coverage": coverage_metrics['total_coverage'],
                    "genre_coverage": coverage_metrics['genre_coverage']
                },
                "final_recommendations": {
                    "precision_score": final_pak_value,
                    "total_coverage": final_coverage['total_coverage'],
                    "genre_coverage": final_coverage['genre_coverage']
                },
                "system_coverage": {
                    "total_coverage": total_metrics['total_coverage'],
                    "genre_coverage": total_metrics['genre_coverage']
                }
            }
            
            # Se specificato un file di esperimento, aggiorna il file con le metriche
            if experiment_file and os.path.exists(experiment_file):
                try:
                    with open(experiment_file, 'r', encoding='utf-8') as f:
                        experiment_data = json.load(f)
                    
                    # Aggiungi le metriche all'esperimento
                    experiment_data['metrics'] = metrics
                    
                    with open(experiment_file, 'w', encoding='utf-8') as f:
                        json.dump(experiment_data, f, ensure_ascii=False, indent=2)
                        
                    print(f"\nMetriche salvate nel file dell'esperimento: {experiment_file}")
                except Exception as e:
                    print(f"Errore nel salvataggio delle metriche nel file dell'esperimento: {e}")
            
            return metrics
        except Exception as e:
            print(f"Errore durante il caricamento dei dati dei film: {e}")
            import traceback
            traceback.print_exc()
            
            # Ritorna metriche di default
            return {
                "precision_at_k": {
                    "precision_score": 0,
                    "total_coverage": 0,
                    "genre_coverage": 0
                },
                "coverage": {
                    "precision_score": 0,
                    "total_coverage": 0,
                    "genre_coverage": 0
                },
                "final_recommendations": {
                    "precision_score": 0,
                    "total_coverage": 0,
                    "genre_coverage": 0
                },
                "system_coverage": {
                    "total_coverage": 0,
                    "genre_coverage": 0
                },
                "error": str(e)
            }
    
    except Exception as e:
        print(f"Errore nel calcolo delle metriche: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "status": "failed"
        }

def add_metrics_to_results(metrics: Dict, output_file: str = "recommendation_results.json") -> bool:
    """
    Aggiunge le metriche calcolate al file di risultati
    
    Args:
        metrics: Dizionario con le metriche calcolate
        output_file: Percorso del file di output
        
    Returns:
        True se l'operazione è riuscita, False altrimenti
    """
    try:
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            # Aggiungi le metriche ai risultati
            results["metrics"] = metrics
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\nMetriche aggiunte con successo al file {output_file}")
            return True
        else:
            print(f"File {output_file} non trovato")
            return False
    
    except Exception as e:
        print(f"Errore nell'aggiunta delle metriche al file: {e}")
        return False 