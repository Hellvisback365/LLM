import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import pandas as pd
import re
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from dotenv import load_dotenv
import random

# Carica variabili d'ambiente
load_dotenv()

# Percorsi assoluti relativi alla root del progetto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RAG_DIR = os.path.join(PROJECT_ROOT, 'data', 'rag')

# Assicurati che la directory RAG esista
os.makedirs(RAG_DIR, exist_ok=True)

class MovieRAG:
    """
    Classe per gestire il Retrieval Augmented Generation per i film
    Implementazione semplificata che non richiede embeddings esterni
    """
    
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.openrouter_api_key,
            temperature=0.7,
            max_tokens=512,
        )
        self.metrics_definitions = self._load_or_generate_metrics_definitions()
        self.movies_df = None
        
    def _clean_descriptions(self, movies_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Corregge errori comuni nelle descrizioni e nei generi dei film
        
        Args:
            movies_list: Lista di dizionari rappresentanti i film
            
        Returns:
            Lista di film con descrizioni corrette
        """
        corrections = {
            "drammatica Roma": "dramma romantico",
            "Dracula: Dead and Loving It": "Dracula: Dead and Loving It (commedia horror)",
            "Heat": "Heat (azione)",
            "Sense and Sensibility": "Sense and Sensibility (dramma romantico)"
        }
        
        for movie in movies_list:
            # Correggi le descrizioni se ci sono errori noti
            if 'description' in movie:
                for error, correction in corrections.items():
                    if error in movie['description']:
                        movie['description'] = movie['description'].replace(error, correction)
            
            # Correggi anche i titoli nei generi se necessario
            if 'genres' in movie:
                for error, correction in corrections.items():
                    if error in str(movie['genres']):
                        movie['genres'] = str(movie['genres']).replace(error, correction)
        
        return movies_list
    
    def initialize_data(self, movies: List[Dict[str, Any]]):
        """Inizializza i dati dei film"""
        self.movies_df = pd.DataFrame(movies)
        
    def load_or_create_vector_store(self, movies: List[Dict[str, Any]], force_recreate: bool = False):
        """
        Versione semplificata che salva i dati dei film senza creare un vector store
        
        Args:
            movies: Lista di dizionari rappresentanti i film
            force_recreate: Ignorato in questa implementazione
        """
        print("Salvando i dati dei film per elaborazione...")
        self.initialize_data(movies)
        
        # Salva i film in un file JSON per utilizzo futuro
        catalog_path = os.path.join(RAG_DIR, 'movies_catalog.json')
        with open(catalog_path, 'w', encoding='utf-8') as f:
            json.dump(movies, f, ensure_ascii=False, indent=2)
    
    def _filter_by_genres(self, genres: List[str]) -> List[Dict[str, Any]]:
        """
        Filtra i film per genere
        
        Args:
            genres: Lista di generi da cercare
            
        Returns:
            Lista di film che corrispondono ai generi specificati
        """
        if self.movies_df is None:
            # Se self.movies_df non è inizializzato, carica i dati dal file
            catalog_path = os.path.join(RAG_DIR, 'movies_catalog.json')
            if os.path.exists(catalog_path):
                with open(catalog_path, 'r', encoding='utf-8') as f:
                    movies = json.load(f)
                self.initialize_data(movies)
            else:
                return []
        
        # Cerca film che contengano almeno uno dei generi specificati
        filtered_movies = []
        for _, movie in self.movies_df.iterrows():
            movie_genres = str(movie['genres']).lower()
            if any(genre.lower() in movie_genres for genre in genres):
                filtered_movies.append(movie.to_dict())
        
        return filtered_movies[:70]  # Limita a 70 risultati
    
    def _filter_by_popularity(self) -> List[Dict[str, Any]]:
        """
        Seleziona un campione di film senza applicare criteri di popolarità.
        Lascia che sia il LLM a determinare quali film sono popolari in base 
        alle sue conoscenze interne.
        
        Returns:
            Lista di film da cui il LLM determinerà i più popolari
        """
        if self.movies_df is None:
            # Se self.movies_df non è inizializzato, carica i dati dal file
            catalog_path = os.path.join(RAG_DIR, 'movies_catalog.json')
            if os.path.exists(catalog_path):
                with open(catalog_path, 'r', encoding='utf-8') as f:
                    movies = json.load(f)
                self.initialize_data(movies)
            else:
                return []
        
        # Non filtriamo per popolarità, restituiamo un campione rappresentativo
        # Restituisci semplicemente i primi 70 film nell'ordine in cui appaiono nel dataset
        # oppure un campione casuale
        random.seed(42)  # Per risultati riproducibili
        
        # Converti il DataFrame in lista di dizionari
        all_movies = self.movies_df.to_dict('records')
        
        # Seleziona 70 film in modo casuale
        if len(all_movies) > 70:
            return random.sample(all_movies, 70)
        else:
            return all_movies
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Implementazione semplificata di ricerca basata su keyword matching
        
        Args:
            query: Query di ricerca
            k: Numero di risultati da restituire
            
        Returns:
            Lista di film più rilevanti per la query
        """
        # Estrai informazioni dalla query
        precision_keywords = ["precision", "accurate", "relevant", "popular"]
        coverage_keywords = ["coverage", "diversity", "diverse", "variety", "different genres"]
        
        # Controlla quale metrica viene menzionata
        if any(keyword in query.lower() for keyword in precision_keywords):
            # Per precision@k, restituisci film popolari
            results = self._filter_by_popularity()
        elif any(keyword in query.lower() for keyword in coverage_keywords):
            # Per coverage, restituisci film di generi diversi
            common_genres = ["Drama", "Comedy", "Action", "Romance", "Thriller", "Sci-Fi", "Adventure", "Crime"]
            results = self._filter_by_genres(common_genres)
        else:
            # Default: mescola entrambi gli approcci
            popular = self._filter_by_popularity()[:k//2]
            diverse = self._filter_by_genres(["Drama", "Comedy", "Action", "Sci-Fi"])[:k//2]
            
            # Unisci mantenendo l'unicità degli ID
            results = popular.copy()
            movie_ids = {movie['movie_id'] for movie in results}
            
            for movie in diverse:
                if movie['movie_id'] not in movie_ids:
                    results.append(movie)
                    movie_ids.add(movie['movie_id'])
                    
                    if len(results) >= k:
                        break
        
        # Limita al numero richiesto
        return results[:k]
    
    def _load_or_generate_metrics_definitions(self) -> Dict[str, str]:
        """
        Carica o genera definizioni delle metriche di raccomandazione
        
        Returns:
            Dizionario con definizioni delle metriche
        """
        metrics_path = os.path.join(RAG_DIR, 'metrics_definitions.json')
        
        # Se il file esiste, caricalo
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        
        # Definizioni complete che combinano entrambe le implementazioni
        metrics_definitions = {
            "precision_at_k": (
                "La metrica Precision@K misura la proporzione di item raccomandati che sono rilevanti. "
                "Nel contesto dei film, rappresenta la frazione di film raccomandati che l'utente valuterebbe "
                "positivamente. Un sistema con alta precision@k tende a raccomandare film che l'utente apprezzerà "
                "con alta probabilità. Formula: (numero di film raccomandati rilevanti) / (numero totale di film raccomandati)."
            ),
            "coverage": (
                "La metrica Coverage misura la proporzione dell'intero catalogo di film che il sistema è in grado "
                "di raccomandare. Rappresenta la diversità del sistema di raccomandazione in termini di ampiezza "
                "delle raccomandazioni. Un sistema con alta coverage esplora meglio lo spazio dei film disponibili "
                "e riduce il rischio di filter bubble. Formula: (numero di film unici raccomandati a tutti gli utenti) / "
                "(numero totale di film nel catalogo)."
            ),
            "accuracy": "Precisione nel suggerire film che corrispondono alle preferenze dell'utente basate sulla sua cronologia di valutazioni. Film con generi o attori simili a quelli che l'utente ha valutato positivamente in passato.",
            "diversity": "Varietà di generi, registi e stili di film nelle raccomandazioni. Film che coprono un'ampia gamma di categorie per espandere gli interessi dell'utente.",
            "novelty": "Film che l'utente probabilmente non conosce o non ha ancora scoperto, ma che potrebbero interessargli. Film meno mainstream o di nicchia che offrono nuove esperienze."
        }
        
        # Salva le definizioni generate
        with open(metrics_path, 'w') as f:
            json.dump(metrics_definitions, f, indent=2)
            
        return metrics_definitions
    
    def generate_metrics_optimized_catalog(self, movies: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
        """
        Genera un catalogo ottimizzato per una specifica metrica
        
        Args:
            movies: Lista di dizionari rappresentanti i film
            metric: Metrica per cui ottimizzare (precision_at_k, coverage, accuracy, diversity, novelty)
            
        Returns:
            Lista di film ottimizzati per la metrica specificata
        """
        # Inizializza i dati se non già fatto
        if self.movies_df is None:
            self.initialize_data(movies)
            
        # Verifica che la metrica sia supportata
        if metric not in self.metrics_definitions and metric not in ["accuracy", "diversity", "novelty"]:
            supported = list(self.metrics_definitions.keys())
            raise ValueError(f"Metrica {metric} non supportata. Usa una tra: {', '.join(supported)}")
            
        if metric == "precision_at_k" or metric == "accuracy":
            # Per precision@k/accuracy, seleziona film popolari (basati sull'ID come semplificazione)
            return self._filter_by_popularity()
        
        elif metric == "coverage" or metric == "diversity":
            # Per coverage/diversity, cerca di avere almeno un film per ogni genere principale
            if metric == "diversity":
                genre_covered = {}
                diverse_movies = []
                
                for movie in movies:
                    genres = str(movie['genres']).split('|')
                    for genre in genres:
                        if genre not in genre_covered:
                            genre_covered[genre] = True
                            diverse_movies.append(movie)
                            break
                    
                    if len(diverse_movies) >= 100:
                        break
                
                return diverse_movies
            else:
                # Per coverage, seleziona film di generi diversi
                common_genres = ["Drama", "Comedy", "Action", "Romance", "Thriller", "Sci-Fi", "Adventure", "Crime"]
                return self._filter_by_genres(common_genres)
                
        elif metric == "novelty":
            # Per novelty, scegli film con ID più alti (ipotizzando che siano film meno mainstream)
            if self.movies_df is None:
                self.initialize_data(movies)
            
            # Ottieni film meno popolari (con ID più alti nell'esempio)
            return self.movies_df.sort_values('movie_id', ascending=False).head(100).to_dict('records')
            
        else:
            # Default fallback
            return movies[:100]
    
    def merge_catalogs(self, precision_catalog: List[Dict[str, Any]], 
                       coverage_catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Unisce i cataloghi ottimizzati per precision@k e coverage
        
        Args:
            precision_catalog: Catalogo ottimizzato per precision@k
            coverage_catalog: Catalogo ottimizzato per coverage
            
        Returns:
            Catalogo unificato bilanciato tra le due metriche
        """
        # Unisci i cataloghi preservando l'ordine
        merged = []
        
        # Dizionario per tenere traccia degli ID già inseriti
        seen_ids = set()
        
        # Alterna tra i due cataloghi prendendo film da entrambi
        for i in range(max(len(precision_catalog), len(coverage_catalog))):
            # Aggiungi dal catalogo precision se disponibile
            if i < len(precision_catalog):
                movie_id = precision_catalog[i]['movie_id']
                if movie_id not in seen_ids:
                    merged.append(precision_catalog[i])
                    seen_ids.add(movie_id)
            
            # Aggiungi dal catalogo coverage se disponibile
            if i < len(coverage_catalog):
                movie_id = coverage_catalog[i]['movie_id']
                if movie_id not in seen_ids:
                    merged.append(coverage_catalog[i])
                    seen_ids.add(movie_id)
        
        return merged
    
    def get_optimized_catalog_for_llm(self, movies: List[Dict[str, Any]], limit: int = 100) -> str:
        """
        Genera un catalogo ottimizzato bilanciando precision@k e coverage da dare in input al LLM
        
        Args:
            movies: Lista di dizionari rappresentanti i film
            limit: Numero massimo di film nel catalogo finale
            
        Returns:
            Stringa JSON con il catalogo ottimizzato
        """
        # Inizializza i dati se non già fatto
        if self.movies_df is None:
            self.initialize_data(movies)
            self.load_or_create_vector_store(movies)
        
        # Genera cataloghi specifici per le metriche
        precision_catalog = self.generate_metrics_optimized_catalog(movies, 'precision_at_k')
        coverage_catalog = self.generate_metrics_optimized_catalog(movies, 'coverage')
        
        # Unisci i cataloghi
        merged_catalog = self.merge_catalogs(precision_catalog, coverage_catalog)
        
        # Limita la dimensione se necessario
        if limit and len(merged_catalog) > limit:
            merged_catalog = merged_catalog[:limit]
        
        # Correggi eventuali errori nelle descrizioni
        merged_catalog = self._clean_descriptions(merged_catalog)
        
        # Converti in JSON
        catalog_json = json.dumps(merged_catalog, ensure_ascii=False)
        
        return catalog_json


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


def calculate_coverage(all_recommended_items: List[List[int]], all_possible_items: List[int]) -> float:
    """
    Calcola la coverage per un set di raccomandazioni
    
    Args:
        all_recommended_items: Lista di liste di ID raccomandati a diversi utenti
        all_possible_items: Lista di tutti gli ID di item possibili nel catalogo
        
    Returns:
        Coverage (0.0 - 1.0)
    """
    if not all_recommended_items or not all_possible_items:
        return 0.0
    
    # Unisci tutte le raccomandazioni in un unico set
    unique_recommended = set()
    for items in all_recommended_items:
        unique_recommended.update(items)
    
    # Calcola la coverage
    coverage = len(unique_recommended) / len(all_possible_items)
    
    return coverage 