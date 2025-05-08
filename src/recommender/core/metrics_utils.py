import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Any

class MetricsCalculator:
    def __init__(self, movies_df: pd.DataFrame, all_available_genres: Set[str] = None):
        self.movies = movies_df
        if self.movies is not None and not self.movies.empty:
            self.all_movie_ids = set(self.movies['movie_id'].unique())
            if all_available_genres is None:
                # Initialize all_available_genres only if movies_df is valid
                if 'genres' in self.movies.columns:
                    self.all_available_genres = set(
                        g for movie_genres in self.movies['genres'].dropna() for g in movie_genres.split('|')
                    )
                else:
                    self.all_available_genres = set() # Handle missing 'genres' column
            else:
                self.all_available_genres = all_available_genres
            
            self.num_total_genres = len(self.all_available_genres) if self.all_available_genres else 0
            self.num_total_movies = len(self.all_movie_ids) if self.all_movie_ids else 0
        else:
            self.all_movie_ids = set()
            self.all_available_genres = set()
            self.num_total_genres = 0
            self.num_total_movies = 0

    def _get_genres_for_ids(self, movie_ids: List[int]) -> Set[str]:
        if self.movies is None or self.movies.empty or not movie_ids or 'genres' not in self.movies.columns:
            return set()
        
        unique_movie_ids = set(movie_ids)
        relevant_movies = self.movies[self.movies['movie_id'].isin(unique_movie_ids)]
        
        genres = set()
        if not relevant_movies.empty:
            for movie_genres_str in relevant_movies['genres'].dropna():
                genres.update(movie_genres_str.split('|'))
        return genres

    def calculate_precision_at_k(self, recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        if not recommended_items or not relevant_items or k <= 0:
            return 0.0
        
        effective_k = min(k, len(recommended_items))
        recommended_k = recommended_items[:effective_k]
        
        relevant_count = sum(1 for item in recommended_k if item in relevant_items)
        return relevant_count / effective_k if effective_k > 0 else 0.0

    def calculate_genre_coverage(self, recommended_ids: List[int]) -> float:
        if not recommended_ids or self.num_total_genres == 0:
            return 0.0
        
        recommended_genres = self._get_genres_for_ids(recommended_ids)
        return len(recommended_genres) / self.num_total_genres if self.num_total_genres > 0 else 0.0

    def calculate_total_item_coverage(self, all_recommended_ids: List[int]) -> float:
        if not all_recommended_ids or self.num_total_movies == 0:
            return 0.0
        unique_recommended = set(all_recommended_ids)
        return len(unique_recommended) / self.num_total_movies if self.num_total_movies > 0 else 0.0

    def calculate_metrics_for_recommendation_set(
        self, 
        recommendations: List[int], 
        relevant_items: List[int], 
        k_values: List[int]
    ) -> Dict[str, Any]:
        """Calculates P@k scores and genre coverage for a single set of recommendations."""
        pak_scores = {
            k: self.calculate_precision_at_k(recommendations, relevant_items, k) for k in k_values
        }
        genre_cov = self.calculate_genre_coverage(recommendations)
        return {"precision_scores": pak_scores, "genre_coverage": genre_cov}

    def compute_all_metrics(
        self,
        metric_results: Dict[int, Dict[str, Dict]], 
        final_evaluation: Dict, 
        per_user_relevant_items: Dict[int, List[int]], 
        k_values: List[int],
        metric_names: List[str]
    ) -> Tuple[Dict[int, Dict[str, Dict]], Dict[str, Dict]]:
        """
        Calculates all per-user and aggregated metrics.
        Returns a tuple: (per_user_calculated_metrics, aggregate_calculated_metrics)
        """
        per_user_calculated_metrics: Dict[int, Dict[str, Dict]] = {}
        
        accumulator_for_aggregation: Dict[str, Dict[str, Any]] = {
            name: {'precision_scores': {k: [] for k in k_values}, 'genre_coverage_scores': []}
            for name in metric_names
        }
        accumulator_for_aggregation['final'] = {
            'precision_scores': {k: [] for k in k_values}, # Will be {k: score} after final calculation
            'genre_coverage_scores': [] # Will be [score] after final calculation
        }

        for user_id, u_metrics_results in metric_results.items():
            user_relevant = per_user_relevant_items.get(user_id, [])
            user_calculated_data: Dict[str, Dict] = {}
            
            for metric_name in metric_names:
                metric_data = u_metrics_results.get(metric_name, {})
                recs = metric_data.get('recommendations', [])
                
                if not recs and self.movies is not None and not self.movies.empty: 
                     print(f"  Utente {user_id}, Metrica {metric_name}: Attenzione - Nessuna raccomandazione trovata.")
                
                calculated_set = self.calculate_metrics_for_recommendation_set(recs, user_relevant, k_values)
                user_calculated_data[metric_name] = calculated_set
                
                for k_val, score in calculated_set["precision_scores"].items():
                    # Ensure k_val exists in the accumulator structure
                    if k_val in accumulator_for_aggregation[metric_name]['precision_scores']:
                        accumulator_for_aggregation[metric_name]['precision_scores'][k_val].append(score)
                accumulator_for_aggregation[metric_name]['genre_coverage_scores'].append(calculated_set["genre_coverage"])
            
            per_user_calculated_metrics[user_id] = user_calculated_data

        all_final_recs = final_evaluation.get('final_recommendations', [])
        all_relevant_items_flat = [
            item for sublist in per_user_relevant_items.values() for item in sublist
        ]
        
        final_pak_scores = {
            k: self.calculate_precision_at_k(all_final_recs, all_relevant_items_flat, k)
            for k in k_values
        }
        final_genre_cov = self.calculate_genre_coverage(all_final_recs)
        
        # Directly assign the already aggregated scores for 'final'
        accumulator_for_aggregation['final']['precision_scores'] = final_pak_scores 
        accumulator_for_aggregation['final']['genre_coverage_scores'] = [final_genre_cov]

        aggregate_calculated_metrics: Dict[str, Dict] = {}
        for name in metric_names + ['final']:
            if name not in accumulator_for_aggregation: continue # Should not happen if logic is correct

            current_metric_aggregation = accumulator_for_aggregation[name]
            
            if name == 'final':
                # P@k for final is already aggregated
                map_at_k_scores = current_metric_aggregation['precision_scores']
            else:
                 map_at_k_scores = {
                    k_val: np.mean(current_metric_aggregation['precision_scores'][k_val]) 
                       if current_metric_aggregation['precision_scores'][k_val] else 0.0
                    for k_val in k_values # Iterate over k_values to ensure all keys are present
                }
            
            avg_gen_cov = np.mean(current_metric_aggregation['genre_coverage_scores']) \
                if current_metric_aggregation['genre_coverage_scores'] else 0.0
            
            key_precision = "precision_scores_agg" if name == 'final' else "map_at_k"
            aggregate_calculated_metrics[name] = {
                key_precision: map_at_k_scores,
                ("genre_coverage" if name == 'final' else "mean_genre_coverage"): avg_gen_cov
            }

        all_recs_flat_for_total_coverage = []
        for uid, u_metrics in metric_results.items():
            for m_name, m_data in u_metrics.items():
                all_recs_flat_for_total_coverage.extend(m_data.get('recommendations', []))
        all_recs_flat_for_total_coverage.extend(all_final_recs)
        
        total_item_cov = self.calculate_total_item_coverage(all_recs_flat_for_total_coverage)
        aggregate_calculated_metrics["total_item_coverage"] = total_item_cov
        
        return per_user_calculated_metrics, aggregate_calculated_metrics 