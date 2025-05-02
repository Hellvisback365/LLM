import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import pandas as pd
import re
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import random
from rank_bm25 import BM25Okapi

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
    
    def __init__(self):
        # Chiave per embeddings OpenAI ufficiale (text-embedding-3-small)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
            print("[WARN] Variabile d'ambiente OPENAI_API_KEY non trovata. Le embeddings potrebbero fallire.")
        self.metrics_definitions = self._load_or_generate_metrics_definitions()
        self.movies_df = None
        # NEW: structures for real retrieval
        self.vector_store = None  # FAISS vector store
        self.bm25 = None          # BM25 index for hybrid / lexical retrieval
        self.corpus_tokens = []
        self.embedding_model = None
        
    def initialize_data(self, movies: List[Dict[str, Any]]):
        """Inizializza i dati dei film"""
        self.movies_df = pd.DataFrame(movies)
        
        # ------------------------------------------------------------------
        # EMBEDDING & INDEX BUILDERS
        # ------------------------------------------------------------------
        # Costruiamo gli indici solo se non esistono già
        if self.vector_store is None:
            self._build_embeddings(movies)
        if self.bm25 is None:
            self._build_bm25(movies)
        
    def _build_embeddings(self, movies: List[Dict[str, Any]], force: bool = False):
        """Costruisce o carica un vector store FAISS con embeddings."""
        if self.vector_store is not None and not force:
            return

        # Persistenza path
        store_path = os.path.join(RAG_DIR, "faiss_store")

        # Inizializza embedding model solo una volta
        if self.embedding_model is None:
            self.embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key
            )

        # Se lo store esiste e non forziamo ricreazione, caricalo
        if os.path.exists(store_path) and not force:
            try:
                # Consenti la deserializzazione perché l'indice è stato creato localmente e può contenere pickle
                self.vector_store = FAISS.load_local(
                    store_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                return
            except Exception as e:
                print(f"Errore nel caricamento dello store FAISS: {e}. Ricreo da zero…")

        # Costruisci da zero
        print("[RAG] Costruzione embeddings per ~{} film".format(len(movies)))
        texts = [f"{m['title']}. Genres: {m['genres'].replace('|', ', ')}" for m in movies]
        metadata = [{"movie_id": m["movie_id"]} for m in movies]

        self.vector_store = FAISS.from_texts(texts, self.embedding_model, metadata)
        # Salva
        self.vector_store.save_local(store_path)

    def _build_bm25(self, movies: List[Dict[str, Any]], force: bool = False):
        """Costruisce indice BM25 per retrieval ibrido"""
        if self.bm25 is not None and not force:
            return

        corpus = [f"{m['title']} {m['genres'].replace('|', ' ')}".lower() for m in movies]
        # tokenize semplice
        self.corpus_tokens = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
    def load_or_create_vector_store(self, movies: List[Dict[str, Any]], force_recreate: bool = False):
        """Carica o crea indice vettoriale + BM25 e salva catalogo."""
        # Inizializza DataFrame interno
        self.initialize_data(movies)
        
        # Costruisci embeddings + FAISS
        self._build_embeddings(movies, force=force_recreate)

        # Costruisci indice BM25 per retrieval ibrido
        self._build_bm25(movies, force=force_recreate)

        # Persisti catalogo per eventuali altri usi
        catalog_path = os.path.join(RAG_DIR, 'movies_catalog.json')
        if not os.path.exists(catalog_path) or force_recreate:
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
        Seleziona un campione di film basandosi esclusivamente sul numero di valutazioni ricevute
        come indicatore reale di popolarità.
        
        Returns:
            Lista dei film più popolari in base al numero di valutazioni
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
        
        # Carica i dati delle valutazioni
        ratings_path = os.path.join(DATA_PROCESSED_DIR, 'filtered_ratings_specific.csv')
        try:
            ratings_df = pd.read_csv(ratings_path)
            
            # Calcola il conteggio delle valutazioni per ogni film
            rating_counts = ratings_df['movie_id'].value_counts().reset_index()
            rating_counts.columns = ['movie_id', 'count']
            
            # Seleziona i top 70 film con più valutazioni
            top_movie_ids = rating_counts.head(70)['movie_id'].tolist()
            
            # Filtra i film per ottenere quelli più popolari
            popular_movies = [m for m in all_movies if int(m['movie_id']) in top_movie_ids]
            
            return popular_movies
        
        except FileNotFoundError:
            print("ATTENZIONE: File delle valutazioni non trovato, usando la selezione casuale come fallback")
            # Fallback alla selezione casuale in caso di errore
            random.seed(42)  # Manteniamo il seed per risultati riproducibili
            
            if len(all_movies) > 70:
                return random.sample(all_movies, 70)
            else:
                return all_movies
    
    # ------------------------------------------------------------------
    # HELPER UTILITIES
    # ------------------------------------------------------------------
    def _get_movies_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        if self.movies_df is None or len(ids) == 0:
            return []
        sub_df = self.movies_df[self.movies_df["movie_id"].isin(ids)]
        return sub_df.to_dict("records")

    # ------------------------------------------------------------------
    # HYBRID SIMILARITY SEARCH + RERANKING
    # ------------------------------------------------------------------
    def similarity_search(self, query: str, k: int = 20, metric_focus: str = "precision_at_k", user_id: int = None) -> List[Dict[str, Any]]:
        """Recupero ibrido (BM25 + FAISS) poi rerank."""

        if self.vector_store is None or self.bm25 is None:
            raise ValueError("Vector/BM25 store non inizializzato. Chiama load_or_create_vector_store prima.")

        # 1) FAISS retrieval
        docs = self.vector_store.similarity_search(query, k=k)
        faiss_ids = [d.metadata["movie_id"] for d in docs]

        # 2) BM25 retrieval
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        # top bm25 indices
        top_idx = np.argsort(bm25_scores)[::-1][:k]
        bm25_ids = [int(self.movies_df.iloc[i]["movie_id"]) for i in top_idx]

        # 3) merge keeping order (simple)
        merged_ids = []
        seen = set()
        for _id in faiss_ids + bm25_ids:
            if _id not in seen:
                merged_ids.append(_id)
                seen.add(_id)
            if len(merged_ids) >= k*2:
                    break
        
        # 4) Rerank secondo la metrica (precision vs coverage) - Ora passiamo user_id
        reranked_ids = self._rerank_by_metric(merged_ids, metric_focus, user_id)

        return self._get_movies_by_ids(reranked_ids[:k])

    def _rerank_by_metric(self, movie_ids: List[int], metric_focus: str, user_id: int = None) -> List[int]:
        """Reranker avanzato: se coverage ordina per generi unici, se precision per rilevanza per utente specifico."""
        if metric_focus == "coverage":
            # preferisci film che introducono nuovi generi
            genre_seen = set()
            scored = []
            for mid in movie_ids:
                movie = self.movies_df[self.movies_df["movie_id"] == mid]
                if movie.empty:
                    continue
                genres = movie.iloc[0]["genres"].split('|')
                new_genres = len([g for g in genres if g not in genre_seen])
                score = new_genres
                scored.append((mid, score))
                genre_seen.update(genres)
            # sort desc by new_genres
            scored.sort(key=lambda x: x[1], reverse=True)
            return [mid for mid, _ in scored]
        else:  # precision@k
            # Se user_id è fornito, usa similarità basata sui contenuti personalizzata per l'utente
            if user_id is not None:
                ratings_path = os.path.join(DATA_PROCESSED_DIR, 'filtered_ratings_specific.csv')
                if os.path.exists(ratings_path):
                    user_ratings = pd.read_csv(ratings_path)
                    user_ratings = user_ratings[user_ratings['user_id'] == user_id]
                    
                    # Ottieni i film piaciuti all'utente (rating >= 4)
                    liked_movie_ids = user_ratings[user_ratings['rating'] >= 4]['movie_id'].tolist()
                    
                    # Se l'utente ha film che gli piacciono, usa quelli per il riordinamento
                    if liked_movie_ids:
                        # Estrai i generi dei film piaciuti
                        liked_genres = set()
                        for mid in liked_movie_ids:
                            movie = self.movies_df[self.movies_df['movie_id'] == mid]
                            if not movie.empty:
                                liked_genres.update(movie.iloc[0]['genres'].split('|'))
                        
                        # Calcola punteggi per i film candidati in base alla sovrapposizione di generi
                        scored = []
                        for mid in movie_ids:
                            movie = self.movies_df[self.movies_df['movie_id'] == mid]
                            if movie.empty:
                                continue
                            
                            # Calcola sovrapposizione di generi
                            movie_genres = set(movie.iloc[0]['genres'].split('|'))
                            overlap = len(movie_genres.intersection(liked_genres))
                            
                            # Calcola popolarità (fattore secondario)
                            popularity = len(user_ratings[user_ratings['movie_id'] == mid])
                            
                            # Punteggio combinato: primario è overlap, secondario è popolarità
                            score = (overlap, popularity)
                            scored.append((mid, score))
                        
                        # Ordina per sovrapposizione (decrescente), poi per popolarità (decrescente)
                        scored.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)
                        return [mid for mid, _ in scored]
            
            # Fallback al ranking basato sulla popolarità generale (comportamento originale)
            ratings_path = os.path.join(DATA_PROCESSED_DIR, 'filtered_ratings_specific.csv')
            if os.path.exists(ratings_path):
                rating_counts = pd.read_csv(ratings_path)['movie_id'].value_counts()
                scored = [(mid, rating_counts.get(mid, 0)) for mid in movie_ids]
                scored.sort(key=lambda x: x[1], reverse=True)
                return [mid for mid, _ in scored]
            return movie_ids  # fallback
    
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
            # Determina i film rilevanti per precision usando rating medio >=4
            ratings_path = os.path.join(DATA_PROCESSED_DIR, 'filtered_ratings_specific.csv')
            ratings_df = pd.read_csv(ratings_path)
            avg_ratings = ratings_df.groupby('movie_id')['rating'].mean()
            relevant_items = avg_ratings[avg_ratings >= 4].index.tolist()
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
        
        # Utilizza retrieval ibrido per entrambe le metriche
        precision_catalog = self.similarity_search("film popolari rilevanti precision", k=limit, metric_focus="precision_at_k")
        coverage_catalog = self.similarity_search("diversi generi film", k=limit, metric_focus="coverage")
        
        merged_catalog = self.merge_catalogs(precision_catalog, coverage_catalog)
        
        if limit and len(merged_catalog) > limit:
            merged_catalog = merged_catalog[:limit]
        
        return json.dumps(merged_catalog, ensure_ascii=False)


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