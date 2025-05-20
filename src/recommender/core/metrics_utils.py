import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import Counter

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
            
            # Per metriche temporali, assicuriamoci di avere gli anni
            if 'year' in self.movies.columns:
                self.movie_release_years = self.movies.set_index('movie_id')['year'].dropna().to_dict()
            else:
                self.movie_release_years = {}
                print("Attenzione: Colonna 'year' non trovata in movies_df. Metriche temporali non saranno accurate.")
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

    def calculate_average_release_year(self, recommended_ids: List[int]) -> float:
        """Calcola l'anno medio di uscita per una lista di film raccomandati."""
        if not recommended_ids or not self.movie_release_years:
            return 0.0
        
        years = [self.movie_release_years.get(mid) for mid in recommended_ids 
                 if mid in self.movie_release_years and pd.notna(self.movie_release_years.get(mid))]
        
        if not years:
            return 0.0 # O np.nan se preferisci gestire NaN a monte
        return np.mean(years)

    def calculate_temporal_dispersion(self, recommended_ids: List[int]) -> float:
        """Calcola la deviazione standard degli anni di uscita per una lista di film raccomandati."""
        if not recommended_ids or not self.movie_release_years or len(recommended_ids) < 2:
             # La deviazione standard non è significativa per meno di 2 item
            return 0.0

        years = [self.movie_release_years.get(mid) for mid in recommended_ids 
                 if mid in self.movie_release_years and pd.notna(self.movie_release_years.get(mid))]
        
        if len(years) < 2:
            return 0.0 # O np.nan
        return np.std(years)

    def calculate_genre_entropy(self, recommended_ids: List[int]) -> float:
        """Calcola l'entropia di Shannon per la distribuzione dei generi nelle raccomandazioni."""
        if not recommended_ids or self.movies is None or self.movies.empty or 'genres' not in self.movies.columns:
            return 0.0

        # Ottieni tutti i generi per i film raccomandati
        genres_list = []
        relevant_movies_subset = self.movies[self.movies['movie_id'].isin(recommended_ids)]
        if not relevant_movies_subset.empty:
            for movie_genres_str in relevant_movies_subset['genres'].dropna():
                genres_list.extend(movie_genres_str.split('|'))
        
        if not genres_list:
            return 0.0

        # Calcola la frequenza di ogni genere
        genre_counts = pd.Series(genres_list).value_counts()
        probabilities = genre_counts / len(genres_list)
        
        # Calcola l'entropia
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

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
        metric_results_for_users: Dict[int, Dict[str, Any]],
        per_user_relevant_items: Dict[int, List[int]],
        k_values: List[int],
        metric_names: List[str], # Nomi delle strategie/metriche e.g., ['precision_at_k', 'coverage']
        experiment_name: Optional[str] = None # Nome dell'esperimento corrente
    ) -> Tuple[Dict[int, Dict[str, Dict]], Dict[str, Dict[str, Any]]]:
        """
        Calcola tutte le metriche per ogni utente e aggrega i risultati.
        Args:
            metric_results_for_users: Risultati grezzi per utente {user_id: {metric_name: {'recommendations': [...], 'explanation': '...'}}}
            per_user_relevant_items: Elementi rilevanti tenuti da parte per ogni utente.
            k_values: Lista di valori K per Precision@k.
            metric_names: Lista dei nomi delle metriche/strategie (es. "precision_at_k", "coverage").
            experiment_name: Nome dell'esperimento corrente, se presente.
        Returns:
            Tuple con (metriche calcolate per utente, metriche aggregate).
        """
        print(f"[MetricsCalculator.compute_all_metrics] Called with: experiment_name='{experiment_name}', metric_names={metric_names}")

        per_user_calculated_metrics: Dict[int, Dict[str, Dict]] = {}
        
        # Inizializza l'accumulatore per ogni metrica/strategia nominata in metric_names
        accumulator_for_aggregation: Dict[str, Dict[str, Any]] = {
            name: {
                'precision_scores': {k: [] for k in k_values}, 
                'genre_coverage_scores': [],
                'average_release_year_scores': [],
                'temporal_dispersion_scores': [],
                'genre_entropy_scores': [],
                'total_recommendations': 0,
                'unique_recommendations': set()
            } for name in metric_names
        }

        for user_id, results_per_metric_name in metric_results_for_users.items():
            user_relevant = set(per_user_relevant_items.get(user_id, []))
            per_user_calculated_metrics[user_id] = {}

            for metric_name in metric_names: # metric_name è es. "precision_at_k" o "coverage"
                if metric_name not in results_per_metric_name:
                    # print(f"Attenzione: Risultati mancanti per la metrica '{metric_name}' per l'utente {user_id}. Salto.")
                    continue
                
                data = results_per_metric_name[metric_name]
                recs = data.get('recommendations', [])
                recs_ids = [m['movie_id'] for m in recs if isinstance(m, dict) and 'movie_id' in m]
                
                # Calcola P@k e Genre Coverage di base
                calculated_set = self.calculate_metrics_for_recommendation_set(recs_ids, user_relevant, k_values)
                print(f"  [MetricsCalculator.compute_all_metrics] User {user_id}, metric_name '{metric_name}', experiment_name '{experiment_name}': Initial calculated_set: {calculated_set.keys()}")
                
                # Calcola metriche specifiche basate su experiment_name e metric_name (chiave del prompt)
                if experiment_name:
                    if experiment_name == "precision_at_k_recency" and metric_name == "precision_at_k":
                        calculated_set["average_release_year"] = self.calculate_average_release_year(recs_ids)
                        accumulator_for_aggregation[metric_name]['average_release_year_scores'].append(calculated_set["average_release_year"])
                        print(f"    Added 'average_release_year': {calculated_set.get('average_release_year')}")
                    
                    elif experiment_name == "coverage_temporal" and metric_name == "coverage":
                        calculated_set["temporal_dispersion"] = self.calculate_temporal_dispersion(recs_ids)
                        accumulator_for_aggregation[metric_name]['temporal_dispersion_scores'].append(calculated_set["temporal_dispersion"])
                        print(f"    Added 'temporal_dispersion': {calculated_set.get('temporal_dispersion')}")
                    
                    elif experiment_name == "coverage_genre_balance" and metric_name == "coverage":
                        calculated_set["genre_entropy"] = self.calculate_genre_entropy(recs_ids)
                        accumulator_for_aggregation[metric_name]['genre_entropy_scores'].append(calculated_set["genre_entropy"])
                        print(f"    Added 'genre_entropy': {calculated_set.get('genre_entropy')}")
                    
                    # Per combined_serendipity_temporal, la parte "temporal" si applica al prompt di "coverage"
                    elif experiment_name == "combined_serendipity_temporal" and metric_name == "coverage":
                        calculated_set["temporal_dispersion"] = self.calculate_temporal_dispersion(recs_ids)
                        accumulator_for_aggregation[metric_name]['temporal_dispersion_scores'].append(calculated_set["temporal_dispersion"])
                        print(f"    Added 'temporal_dispersion' for combined_serendipity_temporal: {calculated_set.get('temporal_dispersion')}")
                
                print(f"  [MetricsCalculator.compute_all_metrics] User {user_id}, metric_name '{metric_name}': Final calculated_set keys: {calculated_set.keys()}")
                per_user_calculated_metrics[user_id][metric_name] = calculated_set
                
                # Accumula per P@k e Genre Coverage standard
                for k in k_values:
                    accumulator_for_aggregation[metric_name]['precision_scores'][k].append(calculated_set['precision_scores'][k])
                accumulator_for_aggregation[metric_name]['genre_coverage_scores'].append(calculated_set['genre_coverage'])
                accumulator_for_aggregation[metric_name]['total_recommendations'] += len(recs_ids)
                accumulator_for_aggregation[metric_name]['unique_recommendations'].update(recs_ids)

        aggregate_calculated_metrics = self.aggregate_calculated_metrics(accumulator_for_aggregation, k_values, metric_names, experiment_name)
        
        print(f"[MetricsCalculator.compute_all_metrics] Returning per_user_calculated_metrics (sample for user {list(per_user_calculated_metrics.keys())[0] if per_user_calculated_metrics else 'N/A'}): {per_user_calculated_metrics.get(list(per_user_calculated_metrics.keys())[0] if per_user_calculated_metrics else None, {}).keys()}")
        print(f"[MetricsCalculator.compute_all_metrics] Returning aggregate_calculated_metrics: {aggregate_calculated_metrics.keys()}")
        
        return per_user_calculated_metrics, aggregate_calculated_metrics

    def aggregate_calculated_metrics(self, accumulator: Dict[str, Dict[str, Any]], k_values: List[int], metric_names: List[str], experiment_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Aggrega le metriche calcolate da tutti gli utenti."""
        print(f"[MetricsCalculator.aggregate_calculated_metrics] Called with: experiment_name='{experiment_name}', metric_names={metric_names}")
        aggregate_results: Dict[str, Dict[str, Any]] = {}
        overall_unique_recs_all_strategies = set()

        for name in metric_names: # name è es. "precision_at_k" o "coverage"
            current_metric_aggregation = accumulator.get(name)
            if not current_metric_aggregation:
                continue

            aggregated_data_for_name: Dict[str, Any] = {
                'map_at_k': {},
                'mean_genre_coverage': 0.0,
                'num_recommendations': current_metric_aggregation['total_recommendations'],
                'num_unique_recommendations': len(current_metric_aggregation['unique_recommendations'])
            }
            overall_unique_recs_all_strategies.update(current_metric_aggregation['unique_recommendations'])

            for k in k_values:
                scores_k = current_metric_aggregation['precision_scores'].get(k, [])
                aggregated_data_for_name['map_at_k'][k] = np.mean(scores_k) if scores_k else 0.0
            
            genre_cov_scores = current_metric_aggregation['genre_coverage_scores']
            aggregated_data_for_name['mean_genre_coverage'] = np.mean(genre_cov_scores) if genre_cov_scores else 0.0

            # Aggrega metriche specifiche basate su experiment_name e name (chiave del prompt)
            print(f"  [MetricsCalculator.aggregate_calculated_metrics] Processing strategy '{name}', experiment_name '{experiment_name}': Initial aggregated_data_for_name keys: {aggregated_data_for_name.keys()}")
            if experiment_name:
                if experiment_name == "precision_at_k_recency" and name == "precision_at_k":
                    avg_year_scores = current_metric_aggregation.get('average_release_year_scores', [])
                    aggregated_data_for_name["avg_release_year"] = np.mean(avg_year_scores) if avg_year_scores else 0.0
                    print(f"    Added 'avg_release_year': {aggregated_data_for_name.get('avg_release_year')}")
                
                elif experiment_name == "coverage_temporal" and name == "coverage":
                    temp_disp_scores = current_metric_aggregation.get('temporal_dispersion_scores', [])
                    aggregated_data_for_name["avg_temporal_dispersion"] = np.mean(temp_disp_scores) if temp_disp_scores else 0.0
                    print(f"    Added 'avg_temporal_dispersion': {aggregated_data_for_name.get('avg_temporal_dispersion')}")
                
                elif experiment_name == "coverage_genre_balance" and name == "coverage":
                    genre_ent_scores = current_metric_aggregation.get('genre_entropy_scores', [])
                    aggregated_data_for_name["avg_genre_entropy"] = np.mean(genre_ent_scores) if genre_ent_scores else 0.0
                    print(f"    Added 'avg_genre_entropy': {aggregated_data_for_name.get('avg_genre_entropy')}")

                elif experiment_name == "combined_serendipity_temporal" and name == "coverage":
                    temp_disp_scores = current_metric_aggregation.get('temporal_dispersion_scores', [])
                    aggregated_data_for_name["avg_temporal_dispersion"] = np.mean(temp_disp_scores) if temp_disp_scores else 0.0
                    print(f"    Added 'avg_temporal_dispersion' for combined_serendipity_temporal: {aggregated_data_for_name.get('avg_temporal_dispersion')}")
            
            print(f"  [MetricsCalculator.aggregate_calculated_metrics] Strategy '{name}': Final aggregated_data_for_name keys: {aggregated_data_for_name.keys()}")
            aggregate_results[name] = aggregated_data_for_name
        
        # Calcola la copertura totale degli item su tutte le strategie e tutti gli utenti
        if self.num_total_movies > 0:
            total_item_coverage = len(overall_unique_recs_all_strategies) / self.num_total_movies
        else:
            total_item_coverage = 0.0
        
        # Potrebbe essere utile aggiungere 'total_item_coverage' a una chiave specifica di aggregate_results
        # o restituirlo separatamente. Per ora lo aggiungo a un livello generale se non ci sono metriche specifiche.
        if not aggregate_results: aggregate_results['overall'] = {} # fallback
        # Mettiamolo in ogni strategia aggregata per ora, o creiamo una sezione 'overall_system_metrics'
        # Per semplicità, lo aggiungo a una chiave "system_wide" se aggregate_results ha elementi, altrimenti "overall"
        target_key_for_item_coverage = list(aggregate_results.keys())[0] if aggregate_results else "overall"
        if target_key_for_item_coverage not in aggregate_results: aggregate_results[target_key_for_item_coverage] = {}
        aggregate_results[target_key_for_item_coverage]['total_item_coverage_system'] = total_item_coverage
            
        return aggregate_results 