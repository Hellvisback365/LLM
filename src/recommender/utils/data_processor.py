import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Costanti per i percorsi file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Assicurati che la directory processed esista
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# Costanti per i nomi delle colonne
RATINGS_COLUMNS = ['user_id', 'movie_id', 'rating', 'timestamp']
MOVIES_COLUMNS = ['movie_id', 'title', 'genres']

def load_ratings(file_path: str = None) -> pd.DataFrame:
    """
    Carica i dati delle valutazioni degli utenti da un file .dat
    
    Args:
        file_path: Il percorso del file, se None usa il percorso predefinito
        
    Returns:
        DataFrame con le valutazioni
    """
    if file_path is None:
        file_path = os.path.join(DATA_RAW_DIR, 'ratings.dat')
    
    # I file .dat del dataset MovieLens sono separati da '::'
    ratings = pd.read_csv(file_path, sep='::', engine='python', header=None, names=RATINGS_COLUMNS)
    return ratings

def load_movies(file_path: str = None) -> pd.DataFrame:
    """
    Carica i dati dei film da un file .dat
    
    Args:
        file_path: Il percorso del file, se None usa il percorso predefinito
        
    Returns:
        DataFrame con i film
    """
    if file_path is None:
        file_path = os.path.join(DATA_RAW_DIR, 'movies.dat')
    
    # I file .dat del dataset MovieLens sono separati da '::'
    movies = pd.read_csv(file_path, sep='::', engine='python', header=None, names=MOVIES_COLUMNS, encoding='latin-1')
    return movies

def filter_users_by_min_ratings(ratings: pd.DataFrame, min_ratings: int = 100) -> pd.DataFrame:
    """
    Filtra gli utenti che hanno meno di min_ratings valutazioni
    
    Args:
        ratings: DataFrame con le valutazioni
        min_ratings: Numero minimo di valutazioni richieste (modificato a 100 per default)
        
    Returns:
        DataFrame con le valutazioni degli utenti che hanno almeno min_ratings valutazioni
    """
    # Conta il numero di valutazioni per utente
    user_counts = ratings['user_id'].value_counts()
    
    # Filtra gli utenti con almeno min_ratings valutazioni
    valid_users = user_counts[user_counts >= min_ratings].index
    
    # Filtra le valutazioni per includere solo gli utenti validi
    filtered_ratings = ratings[ratings['user_id'].isin(valid_users)]
    
    print(f"Filtraggio utenti: {len(valid_users)}/{len(user_counts)} utenti mantengono almeno {min_ratings} valutazioni.")
    print(f"Valutazioni rimanenti: {len(filtered_ratings)}/{len(ratings)} ({len(filtered_ratings)/len(ratings)*100:.2f}%)")
    
    return filtered_ratings

def filter_users_by_specific_users(ratings: pd.DataFrame, user_ids: list = [1, 2]) -> pd.DataFrame:
    """
    Filtra solo per includere specifici utenti
    
    Args:
        ratings: DataFrame con le valutazioni
        user_ids: Lista di ID utente da mantenere (default: utenti 1 e 2)
        
    Returns:
        DataFrame con le valutazioni degli utenti specificati
    """
    # Filtra le valutazioni per includere solo gli utenti selezionati
    filtered_ratings = ratings[ratings['user_id'].isin(user_ids)]
    
    print(f"Filtraggio utenti: selezionati {len(user_ids)} utenti specifici (IDs: {user_ids}).")
    print(f"Valutazioni rimanenti: {len(filtered_ratings)}/{len(ratings)} ({len(filtered_ratings)/len(ratings)*100:.2f}%)")
    
    return filtered_ratings

def create_user_profiles(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Crea profili utente categorizzando le valutazioni come 'like' (4-5) o 'dislike' (1-3)
    
    Args:
        ratings: DataFrame con le valutazioni
        
    Returns:
        DataFrame con profili utente (film piaciuti e non piaciuti)
    """
    # Crea una copia per non modificare l'originale
    df = ratings.copy()
    
    # Aggiungi una colonna che indica se il film è piaciuto o meno
    df['preference'] = df['rating'].apply(lambda x: 'like' if x >= 4 else 'dislike')
    
    # Crea un DataFrame per i profili utente
    user_profiles = pd.DataFrame(index=df['user_id'].unique())
    
    # Aggiungi liste di film piaciuti e non piaciuti
    liked_movies = df[df['preference'] == 'like'].groupby('user_id')['movie_id'].apply(list)
    disliked_movies = df[df['preference'] == 'dislike'].groupby('user_id')['movie_id'].apply(list)
    
    user_profiles['liked_movies'] = liked_movies
    user_profiles['disliked_movies'] = disliked_movies
    
    # Riempi i valori NaN con liste vuote
    user_profiles['liked_movies'] = user_profiles['liked_movies'].apply(lambda x: x if isinstance(x, list) else [])
    user_profiles['disliked_movies'] = user_profiles['disliked_movies'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Aggiungi conteggi
    user_profiles['num_liked'] = user_profiles['liked_movies'].apply(len)
    user_profiles['num_disliked'] = user_profiles['disliked_movies'].apply(len)
    user_profiles['total_ratings'] = user_profiles['num_liked'] + user_profiles['num_disliked']
    
    return user_profiles

def process_dataset():
    """
    Processo completo di caricamento e elaborazione del dataset
    
    Returns:
        Tuple con (ratings filtrati, profili utente, film)
    """
    try:
        # Carica i dati
        ratings = load_ratings()
        movies = load_movies()
        
        # Filtra gli utenti con meno di 100 valutazioni
        filtered_ratings = filter_users_by_min_ratings(ratings)
        
        # Crea profili utente
        user_profiles = create_user_profiles(filtered_ratings)
        
        # Salva i dati elaborati
        filtered_ratings.to_csv(os.path.join(DATA_PROCESSED_DIR, 'filtered_ratings.csv'), index=False)
        user_profiles.to_csv(os.path.join(DATA_PROCESSED_DIR, 'user_profiles.csv'))
        movies.to_csv(os.path.join(DATA_PROCESSED_DIR, 'movies.csv'), index=False)
        
        # Crea anche una versione JSON del catalogo film per l'input al modello
        movies_json = movies.to_json(orient='records')
        with open(os.path.join(DATA_PROCESSED_DIR, 'movies_catalog.json'), 'w') as f:
            f.write(movies_json)
        
        print(f"Processamento completato. Dati salvati in {DATA_PROCESSED_DIR}")
        
        return filtered_ratings, user_profiles, movies
        
    except Exception as e:
        print(f"Errore durante il processamento del dataset: {e}")
        raise

def get_movie_catalog_for_llm(limit: int = None) -> str:
    """
    Restituisce il catalogo dei film in formato JSON per l'input al modello LLM
    
    Args:
        limit: Numero massimo di film da includere, se None include tutti
        
    Returns:
        Stringa JSON con i dati dei film
    """
    try:
        # Carica i film dal file elaborato se esiste
        catalog_path = os.path.join(DATA_PROCESSED_DIR, 'movies_catalog.json')
        
        if os.path.exists(catalog_path):
            with open(catalog_path, 'r') as f:
                catalog_json = f.read()
                
            # Se è richiesto un limite, limita il numero di film
            if limit:
                import json
                catalog = json.loads(catalog_json)
                catalog = catalog[:limit]
                catalog_json = json.dumps(catalog)
            
            return catalog_json
        
        # Se il file non esiste, carica e processa i dati
        movies = load_movies()
        
        # Seleziona un sottoinsieme di film se richiesto
        if limit:
            movies = movies.head(limit)
        
        # Converti in JSON
        import json
        catalog_json = movies.to_json(orient='records')
        
        return catalog_json
    
    except Exception as e:
        print(f"Errore durante l'ottenimento del catalogo film: {e}")
        return "[]"  # Restituisci un JSON vuoto in caso di errore 