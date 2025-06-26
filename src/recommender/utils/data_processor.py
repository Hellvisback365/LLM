import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import math # Aggiunto per ceil

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
    movies = pd.read_csv(file_path, sep='::', engine='python', header=None, names=['movie_id', 'title_with_year', 'genres'], encoding='latin-1')
    
    # Estrai l'anno dal titolo e crea una nuova colonna 'year'
    # Regex per trovare (YYYY) alla fine della stringa del titolo
    # Assicurarsi che 'title_with_year' sia trattato come stringa
    movies['title_with_year'] = movies['title_with_year'].astype(str)
    year_extraction = movies['title_with_year'].str.extract(r'\((\d{4})\)$', expand=False)
    
    # Converti in numerico, errori a NaT/NaN che poi possono essere gestiti (es. fillna o dropna)
    movies['year'] = pd.to_numeric(year_extraction, errors='coerce')
    
    # Opzionale: creare una colonna titolo senza l'anno
    # movies['title'] = movies['title_with_year'].str.replace(r'\\s*\\(\\d{4}\\)$', '', regex=True).str.strip()
    # Per ora, rinominiamo title_with_year in title per coerenza se altre parti del codice usano 'title'
    movies.rename(columns={'title_with_year': 'title'}, inplace=True)

    print(f"Film caricati: {len(movies)}")
    # Stampa un campione di titoli e anni estratti per verifica
    # print("Esempio estrazione anno:")
    # print(movies[['title', 'year']].head())
    # print(f"Anni non parsabili (NaN): {movies['year'].isna().sum()}")

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

def filter_users_by_specific_users(ratings: pd.DataFrame, user_ids: list = None) -> pd.DataFrame:
    """
    Filtra solo per includere specifici utenti
    
    Args:
        ratings: DataFrame con le valutazioni
        user_ids: Lista di ID utente da mantenere (default: utenti 1 e 2)
        
    Returns:
        DataFrame con le valutazioni degli utenti specificati
    """
    if user_ids is None:
        print("Nessun ID utente specifico fornito, verranno utilizzati tutti gli utenti.")
        return ratings
        
    # Filtra le valutazioni per includere solo gli utenti selezionati
    filtered_ratings = ratings[ratings['user_id'].isin(user_ids)]
    
    print(f"Filtraggio utenti: selezionati {len(user_ids)} utenti specifici (IDs: {user_ids}).")
    print(f"Valutazioni rimanenti: {len(filtered_ratings)}/{len(ratings)} ({len(filtered_ratings)/len(ratings)*100:.2f}%)")
    
    return filtered_ratings

def create_user_profiles(ratings: pd.DataFrame, hold_out_percentage: float = 0.2) -> pd.DataFrame:
    """
    Crea profili utente separando una percentuale di film piaciuti per la valutazione (hold-out).
    
    Args:
        ratings: DataFrame con le valutazioni.
        hold_out_percentage: Percentuale di film piaciuti da tenere come hold-out (default: 0.2).
        
    Returns:
        DataFrame con profili utente:
        - 'profile_liked_movies': Film piaciuti da usare nel profilo (1 - hold_out_percentage).
        - 'disliked_movies': Film non piaciuti (tutti).
        - 'held_out_liked_movies': Film piaciuti nascosti per la valutazione (ground truth).
        - Conteggi relativi.
    """
    if not (0 <= hold_out_percentage < 1):
        raise ValueError("hold_out_percentage deve essere tra 0 (incluso) e 1 (escluso).")
        
    df = ratings.copy()
    df['preference'] = df['rating'].apply(lambda x: 'like' if x >= 4 else 'dislike')
    
    user_profiles_data = []
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id]
        
        all_liked_movies = user_data[user_data['preference'] == 'like']['movie_id'].tolist()
        disliked_movies = user_data[user_data['preference'] == 'dislike']['movie_id'].tolist()
        
        profile_liked_movies = []
        held_out_liked_movies = []
        
        if all_liked_movies:
            np.random.shuffle(all_liked_movies) # Mescola per casualità
            num_hold_out = math.ceil(len(all_liked_movies) * hold_out_percentage)
            # Assicurati che num_hold_out non sia maggiore del numero di film piaciuti
            num_hold_out = min(num_hold_out, len(all_liked_movies))

            # Separa gli ID
            held_out_liked_movies = all_liked_movies[:num_hold_out]
            profile_liked_movies = all_liked_movies[num_hold_out:]
            
        user_profiles_data.append({
            'user_id': user_id,
            'profile_liked_movies': profile_liked_movies,
            'disliked_movies': disliked_movies,
            'held_out_liked_movies': held_out_liked_movies,
            'num_profile_liked': len(profile_liked_movies),
            'num_disliked': len(disliked_movies),
            'num_held_out': len(held_out_liked_movies),
            'total_ratings_in_profile': len(profile_liked_movies) + len(disliked_movies)
        })
        
    user_profiles = pd.DataFrame(user_profiles_data)
    user_profiles = user_profiles.set_index('user_id') # Imposta user_id come indice
    
    print(f"Creati profili per {len(user_profiles)} utenti. "
          f"Hold-out {hold_out_percentage*100:.0f}% dei film piaciuti.")
          
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