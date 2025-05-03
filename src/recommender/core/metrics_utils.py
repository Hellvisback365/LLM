from typing import List, Dict
import pandas as pd


def calculate_precision_at_k(recommended_items: List[int], relevant_items: List[int], k: int = None) -> float:
    """
    Calcola la precision@k per un set di raccomandazioni
    
    Args:
        recommended_items: Lista di ID di item raccomandati
        relevant_items: Lista di ID di item rilevanti
        k: Numero di item da considerare (default: lunghezza di recommended_items)
    
    Returns:
        Precision@k (0.0 - 1.0)
    """
    if not recommended_items or not relevant_items:
        return 0.0
    
    if k is None:
        k = len(recommended_items)
    else:
        k = min(k, len(recommended_items))
    
    # Considera solo i primi k item raccomandati
    recommended_k = recommended_items[:k]
    
    # Calcola quanti degli item raccomandati sono rilevanti
    relevant_count = sum(1 for item in recommended_k if item in relevant_items)
    
    # Calcola la precision@k
    return relevant_count / k if k > 0 else 0.0


def calculate_coverage(recommended_items: List[int], movies_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcola metriche di coverage complete per un set di raccomandazioni
    
    Args:
        recommended_items: Lista di ID di film raccomandati
        movies_df: DataFrame contenente i dati dei film (deve avere colonne 'movie_id' e 'genres')
    
    Returns:
        Dict con due metriche di coverage:
        - total_coverage: proporzione di film unici raccomandati rispetto al catalogo totale
        - genre_coverage: proporzione di generi unici coperti rispetto a tutti i generi disponibili
    """
    if not recommended_items or movies_df.empty:
        return {"total_coverage": 0.0, "genre_coverage": 0.0}
    
    # Coverage totale
    total_movies = len(movies_df)
    unique_recommended = set(recommended_items)
    total_coverage = len(unique_recommended) / total_movies
    
    # Coverage dei generi
    all_genres = set()
    recommended_genres = set()
    
    # Estrai tutti i generi disponibili
    for genres in movies_df['genres']:
        if isinstance(genres, str):
            all_genres.update(genres.split('|'))
    
    # Estrai i generi delle raccomandazioni
    for movie_id in unique_recommended:
        movie = movies_df[movies_df['movie_id'] == movie_id]
        if not movie.empty and isinstance(movie.iloc[0]['genres'], str):
            recommended_genres.update(movie.iloc[0]['genres'].split('|'))
    
    genre_coverage = len(recommended_genres) / len(all_genres) if all_genres else 0.0
    
    return {
        "total_coverage": total_coverage,
        "genre_coverage": genre_coverage
    } 