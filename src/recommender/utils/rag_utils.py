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
    
    def _extract_all_genres(self) -> List[str]:
        """
        Estrae automaticamente tutti i generi presenti nel dataset
        
        Returns:
            Lista di tutti i generi unici presenti nel dataset
        """
        if self.movies_df is None:
            catalog_path = os.path.join(RAG_DIR, 'movies_catalog.json')
            if os.path.exists(catalog_path):
                with open(catalog_path, 'r', encoding='utf-8') as f:
                    movies = json.load(f)
                self.initialize_data(movies)
            else:
                return []
        
        # Estrai tutti i generi unici dal dataset
        all_genres = set()
        for _, movie in self.movies_df.iterrows():
            genres = str(movie['genres']).split('|')
            for genre in genres:
                if genre and genre.strip():  # Ignora stringhe vuote
                    all_genres.add(genre.strip())
        
        return list(all_genres)

    def _filter_by_genres(self, genres: List[str] = None) -> List[Dict[str, Any]]:
        """
        Seleziona un campione rappresentativo di film che offre una buona coverage dei generi.
        Invece di applicare filtri rigidi, fornisce un ampio campione al LLM
        lasciandogli la libertà di selezionare i film più appropriati per la metrica coverage.
        
        Args:
            genres: Lista di generi da considerare (opzionale, se None usa tutti i generi disponibili)
            
        Returns:
            Lista di film rappresentativa per la metrica coverage
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
        
        # Converti il DataFrame in lista di dizionari
        all_movies = self.movies_df.to_dict('records')
        
        # Se non sono specificati generi, usa tutti i generi disponibili nel dataset
        if genres is None or len(genres) == 0:
            genres = self._extract_all_genres()
        
        # Strategia: Creare un campione che massimizzi la diversità di generi
        # senza limitare artificialmente le scelte del LLM
        
        # Inizializza un campione vuoto e un set di ID già inclusi
        sample = []
        included_ids = set()
        
        # Per ogni genere, includi alcuni film rappresentativi
        for genre in genres:
            genre_lower = genre.lower()
            genre_movies = []
            
            # Trova film di questo genere
            for movie in all_movies:
                if movie['movie_id'] not in included_ids and genre_lower in str(movie['genres']).lower():
                    genre_movies.append(movie)
                    
                    # Limita a 5 film per genere per garantire diversità
                    if len(genre_movies) >= 5:
                        break
            
            # Aggiungi questi film al campione e aggiorna gli ID inclusi
            for movie in genre_movies:
                sample.append(movie)
                included_ids.add(movie['movie_id'])
        
        # Se non abbiamo abbastanza film, aggiungi altri film casuali fino a raggiungere una dimensione adeguata
        if len(sample) < 100 and len(all_movies) > len(sample):
            # Film rimanenti non ancora inclusi nel campione
            remaining_movies = [movie for movie in all_movies if movie['movie_id'] not in included_ids]
            
            # Aggiungi film casuali (con seed fisso per riproducibilità)
            random.seed(42)
            remaining_to_add = min(100 - len(sample), len(remaining_movies))
            if remaining_to_add > 0:
                sample.extend(random.sample(remaining_movies, remaining_to_add))
        
        # Limita la dimensione del campione per non sovraccaricare il LLM
        return sample[:150]  # Ampliamo il limite per dare più opzioni al LLM
    
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
        Implementazione che fornisce dati ricchi al LLM per permettergli 
        di applicare autonomamente le metriche di coverage e precision@k.
        Non impone filtri rigidi ma offre una selezione ampia e diversificata.
        
        Args:
            query: Query di ricerca
            k: Numero di risultati da restituire
            
        Returns:
            Lista di film rilevanti per la query
        """
        # Prepariamo due selezioni complementari per le due metriche
        
        # Per precision@k: selezione che privilegia la popolarità
        precision_selection = self._filter_by_popularity()
        
        # Per coverage: selezione che privilegia la diversità dei generi
        # Non specifichiamo manualmente i generi, ma usiamo tutti quelli disponibili
        coverage_selection = self._filter_by_genres()
        
        # Estraiamo parole chiave dalla query per capire su quale metrica concentrarsi
        precision_keywords = ["precision", "accurate", "relevant", "popular", "liked", "rating"]
        coverage_keywords = ["coverage", "diversity", "diverse", "variety", "different", "genres", "broad"]
        
        # Prepara una risposta che combina entrambe le selezioni
        # dando la priorità a una delle due in base alla query
        combined_selection = []
        existing_ids = set()
        
        # Determiniamo le proporzioni in base alla query
        precision_focus = any(keyword in query.lower() for keyword in precision_keywords)
        coverage_focus = any(keyword in query.lower() for keyword in coverage_keywords)
        
        # Se entrambe le metriche sono menzionate o nessuna è menzionata,
        # facciamo un mix bilanciato
        if (precision_focus and coverage_focus) or (not precision_focus and not coverage_focus):
            # Alterna tra le due selezioni
            for i in range(max(len(precision_selection), len(coverage_selection))):
                # Aggiungi dalla selezione precision
                if i < len(precision_selection) and precision_selection[i]['movie_id'] not in existing_ids:
                    combined_selection.append(precision_selection[i])
                    existing_ids.add(precision_selection[i]['movie_id'])
                
                # Aggiungi dalla selezione coverage
                if i < len(coverage_selection) and coverage_selection[i]['movie_id'] not in existing_ids:
                    combined_selection.append(coverage_selection[i])
                    existing_ids.add(coverage_selection[i]['movie_id'])
                
                # Limita la dimensione
                if len(combined_selection) >= k*20:
                    break
        
        # Se la query si concentra sulla precision
        elif precision_focus:
            # Prima aggiungiamo tutti i film dalla selezione precision
            for movie in precision_selection:
                if movie['movie_id'] not in existing_ids:
                    combined_selection.append(movie)
                    existing_ids.add(movie['movie_id'])
            
            # Poi aggiungiamo alcuni film dalla selezione coverage
            coverage_added = 0
            for movie in coverage_selection:
                if movie['movie_id'] not in existing_ids and coverage_added < 50:
                    combined_selection.append(movie)
                    existing_ids.add(movie['movie_id'])
                    coverage_added += 1
        
        # Se la query si concentra sulla coverage
        elif coverage_focus:
            # Prima aggiungiamo tutti i film dalla selezione coverage
            for movie in coverage_selection:
                if movie['movie_id'] not in existing_ids:
                    combined_selection.append(movie)
                    existing_ids.add(movie['movie_id'])
            
            # Poi aggiungiamo alcuni film dalla selezione precision
            precision_added = 0
            for movie in precision_selection:
                if movie['movie_id'] not in existing_ids and precision_added < 50:
                    combined_selection.append(movie)
                    existing_ids.add(movie['movie_id'])
                    precision_added += 1
        
        # Restituiamo un campione più ampio per dare libertà al LLM
        # Il parametro k è moltiplicato per dare al LLM più opzioni
        return combined_selection[:k*20]
    
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
        
        # Definizioni solo per le due metriche richieste
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
            )
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
            metric: Metrica per cui ottimizzare (precision_at_k o coverage)
            
        Returns:
            Lista di film ottimizzati per la metrica specificata
        """
        # Inizializza i dati se non già fatto
        if self.movies_df is None:
            self.initialize_data(movies)
            
        # Verifica che la metrica sia supportata
        if metric not in self.metrics_definitions:
            supported = list(self.metrics_definitions.keys())
            raise ValueError(f"Metrica {metric} non supportata. Usa una tra: {', '.join(supported)}")
            
        if metric == "precision_at_k":
            # Per precision@k, usa la selezione che privilegia la popolarità
            return self._filter_by_popularity()
        
        elif metric == "coverage":
            # Per coverage, usa la selezione che massimizza la diversità di generi
            # Nota: non specifichiamo generi manualmente, usiamo quelli estratti dal dataset
            return self._filter_by_genres()
        
        else:
            # Default fallback (non dovrebbe mai verificarsi con la validazione sopra)
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