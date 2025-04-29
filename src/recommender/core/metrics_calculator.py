import os
import json
import pandas as pd
from typing import Dict, List, Any, Set
from src.recommender.utils.rag_utils import calculate_precision_at_k, calculate_coverage

def calculate_metrics_for_recommendations(metric_results: Dict, final_evaluation: Dict) -> Dict:
    """
    Calcola metriche quantitative per le raccomandazioni
    
    Args:
        metric_results: Risultati intermedi delle raccomandazioni per ogni metrica
        final_evaluation: Valutazione finale
        
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
            movies_path = os.path.join('data', 'processed', 'movies_catalog.json')
            with open(movies_path, 'r', encoding='utf-8') as f:
                movies_data = json.load(f)
            
            # Converti in DataFrame
            movies = pd.DataFrame(movies_data)
            
            # Prepara i dati per il calcolo delle metriche
            all_movie_ids = movies['movie_id'].tolist() if 'movie_id' in movies.columns else []
            
            # Simula dati rilevanti per precision@k (top 100 film più popolari come proxy)
            # In un caso reale, questi sarebbero film che l'utente ha già valutato positivamente
            relevant_items = all_movie_ids[:100] if len(all_movie_ids) >= 100 else all_movie_ids
            
            # Calcola precision@k
            precision_pak_value = calculate_precision_at_k(precision_at_k_recs, relevant_items)
            coverage_pak_value = calculate_precision_at_k(coverage_recs, relevant_items)
            final_pak_value = calculate_precision_at_k(final_recs, relevant_items)
            
            print(f"\nPrecision@k per precision_at_k: {precision_pak_value:.4f}")
            print(f"Precision@k per coverage: {coverage_pak_value:.4f}")
            print(f"Precision@k per raccomandazioni finali: {final_pak_value:.4f}")
            
            # Calcola coverage
            all_recommendations = [precision_at_k_recs, coverage_recs, final_recs]
            
            # Per la coverage, calcoliamo il numero di generi unici coperti
            all_genres = set()
            genre_coverage = {}
            
            # Funzione per estrarre i generi di un film
            def get_film_genres(movie_id: int) -> Set[str]:
                movie = movies[movies['movie_id'] == movie_id]
                if not movie.empty and 'genres' in movie.columns:
                    genres_str = movie.iloc[0]['genres']
                    if genres_str and isinstance(genres_str, str):
                        return set(genres_str.split('|'))
                return set()
            
            # Calcola i generi unici per ogni set di raccomandazioni
            for name, recs in [("precision_at_k", precision_at_k_recs), 
                              ("coverage", coverage_recs), 
                              ("final", final_recs)]:
                recs_genres = set()
                for movie_id in recs:
                    recs_genres.update(get_film_genres(movie_id))
                
                all_genres.update(recs_genres)
                genre_coverage[name] = len(recs_genres) / (len(all_genres) if all_genres else 1)
            
            print(f"\nGeneri unici totali identificati: {len(all_genres)}")
            print(f"Generi trovati: {all_genres}")
            
            print(f"\nCoverage per genere (generi unici coperti / totale generi):")
            print(f"  precision_at_k: {genre_coverage.get('precision_at_k', 0):.4f}")
            print(f"  coverage: {genre_coverage.get('coverage', 0):.4f}")
            print(f"  final: {genre_coverage.get('final', 0):.4f}")
            
            # Calcola la coverage totale (film unici raccomandati / totale film)
            all_recommended_ids = set()
            for recs in all_recommendations:
                all_recommended_ids.update(recs)
                
            total_coverage = len(all_recommended_ids) / len(all_movie_ids) if all_movie_ids else 0
            print(f"\nCoverage totale (film unici raccomandati / totale film): {total_coverage:.4f}")
            
            # Crea il dizionario con tutte le metriche
            metrics = {
                "precision_at_k": {
                    "precision_score": precision_pak_value,
                    "genre_coverage": genre_coverage.get('precision_at_k', 0)
                },
                "coverage": {
                    "precision_score": coverage_pak_value,
                    "genre_coverage": genre_coverage.get('coverage', 0)
                },
                "final_recommendations": {
                    "precision_score": final_pak_value,
                    "genre_coverage": genre_coverage.get('final', 0)
                },
                "total_coverage": total_coverage,
                "genre_information": {
                    "total_genres": len(all_genres),
                    "all_genres": list(all_genres)
                }
            }
            
            return metrics
        except Exception as e:
            print(f"Errore durante il caricamento dei dati dei film: {e}")
            import traceback
            traceback.print_exc()
            
            # Ritorna metriche di default
            return {
                "precision_at_k": {
                    "precision_score": 0,
                    "genre_coverage": 0
                },
                "coverage": {
                    "precision_score": 0,
                    "genre_coverage": 0
                },
                "final_recommendations": {
                    "precision_score": 0,
                    "genre_coverage": 0
                },
                "total_coverage": 0,
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