
"""
Sistema di raccomandazione multi-metrica.

Questo modulo implementa il sistema di raccomandazione che utilizza LLM per generare
raccomandazioni ottimizzate per diverse metriche.
"""

import os
import json
import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple

from src.recommender.utils.data_processor import (
    process_dataset, 
    get_movie_catalog_for_llm, 
    filter_users_by_specific_users, 
    load_ratings, 
    load_movies, 
    create_user_profiles
)
from src.recommender.utils.rag_utils import MovieRAG
from src.recommender.api.llm_service import LLMService
from src.recommender.core.metrics_calculator import calculate_metrics_for_recommendations, add_metrics_to_results

class RecommenderSystem:
    """
    Sistema di raccomandazione che ottimizza per diverse metriche.
    """
    
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        """
        Inizializza il sistema di raccomandazione.
        
        Args:
            model_name: Nome del modello LLM da utilizzare
        """
        self.model_name = model_name
        self.llm_service = LLMService(model_name=model_name)
        self.datasets_loaded = False
        self.filtered_ratings = None
        self.user_profiles = None 
        self.movies = None
        self.rag = None
        
    def load_datasets(self, force_reload: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Carica e prepara i dataset.
        
        Args:
            force_reload: Se True, ricarica i dataset anche se già caricati
            
        Returns:
            Tuple con (filtered_ratings, user_profiles, movies)
        """
        if not self.datasets_loaded or force_reload:
            print("\n=== Caricamento e processamento dei dataset ===\n")
            try:
                # Verifica se i file elaborati esistono
                processed_dir = os.path.join('data', 'processed')
                if not force_reload and all(os.path.exists(os.path.join(processed_dir, f)) 
                                    for f in ['filtered_ratings_specific.csv', 'user_profiles_specific.csv', 'movies.csv']):
                    print("Caricamento dati da file elaborati con utenti specifici...")
                    self.filtered_ratings = pd.read_csv(os.path.join(processed_dir, 'filtered_ratings_specific.csv'))
                    self.user_profiles = pd.read_csv(os.path.join(processed_dir, 'user_profiles_specific.csv'), index_col=0)
                    self.movies = pd.read_csv(os.path.join(processed_dir, 'movies.csv'))
                else:
                    print("Elaborazione dati dal dataset grezzo...")
                    # Carica i dati
                    ratings = load_ratings()
                    self.movies = load_movies()
                    
                    # Filtra solo utenti con ID 1 e 2
                    self.filtered_ratings = filter_users_by_specific_users(ratings, [1, 2])
                    
                    # Crea profili utente
                    self.user_profiles = create_user_profiles(self.filtered_ratings)
                    
                    # Salva i dati elaborati
                    os.makedirs(processed_dir, exist_ok=True)
                    self.filtered_ratings.to_csv(os.path.join(processed_dir, 'filtered_ratings_specific.csv'), index=False)
                    self.user_profiles.to_csv(os.path.join(processed_dir, 'user_profiles_specific.csv'))
                    self.movies.to_csv(os.path.join(processed_dir, 'movies.csv'), index=False)
                
                print(f"Dataset processato con successo. {len(self.movies)} film, {len(self.user_profiles)} profili utente specifici (ID: 1, 2).")
                self.datasets_loaded = True
                
                # Prepara il catalogo ottimizzato per il LLM
                print("\n=== Preparazione del catalogo ottimizzato per RAG ===\n")
                self.rag = MovieRAG(model_name=self.model_name)
                
                # Converti movies DataFrame in lista di dizionari
                movies_list = self.movies.to_dict('records')
                
                # Inizializza il vector store
                self.rag.load_or_create_vector_store(movies_list, force_recreate=force_reload)
                
                # Genera e salva il catalogo ottimizzato
                catalog_json = self.rag.get_optimized_catalog_for_llm(movies_list)
                
                # Salva il catalogo ottimizzato
                catalog_path = os.path.join('data', 'processed', 'optimized_catalog.json')
                os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
                with open(catalog_path, 'w', encoding='utf-8') as f:
                    f.write(catalog_json)
                    
                print(f"Catalogo ottimizzato salvato in {catalog_path}")
                
                return self.filtered_ratings, self.user_profiles, self.movies
                
            except Exception as e:
                print(f"Errore durante il caricamento dei dataset: {e}")
                raise
        else:
            print("Dataset già caricati.")
            return self.filtered_ratings, self.user_profiles, self.movies
    
    def get_optimized_catalog(self, limit: int = 100) -> str:
        """
        Ottiene il catalogo ottimizzato per l'LLM.
        
        Args:
            limit: Numero massimo di film da includere
            
        Returns:
            Catalogo ottimizzato in formato JSON
        """
        catalog_path = os.path.join('data', 'processed', 'optimized_catalog.json')
        
        if os.path.exists(catalog_path):
            with open(catalog_path, 'r', encoding='utf-8') as f:
                catalog_json = f.read()
                
            # Limita il numero di film se necessario
            if limit:
                catalog = json.loads(catalog_json)
                catalog = catalog[:limit]
                return json.dumps(catalog, ensure_ascii=False)
            
            return catalog_json
        else:
            print("Catalogo ottimizzato non trovato. Eseguendo il processamento dei dataset...")
            self.load_datasets(force_reload=True)
            return self.get_optimized_catalog(limit)
    
    async def run_metric_recommenders(self, user_id: int = 1) -> Dict:
        """
        Esegue i raccomandatori per le diverse metriche.
        
        Args:
            user_id: ID dell'utente per cui generare raccomandazioni
            
        Returns:
            Dizionario con i risultati per ogni metrica
        """
        print("\n=== Esecuzione dei raccomandatori per metriche specifiche ===\n")
        
        # Assicurati che i dataset siano caricati
        if not self.datasets_loaded:
            self.load_datasets()
        
        # Ottieni il catalogo ottimizzato
        catalog = self.get_optimized_catalog()
        
        # Ottieni il profilo utente
        user_profile = self.user_profiles.loc[user_id].to_dict()
        user_profile_str = json.dumps(user_profile, ensure_ascii=False)
        
        # Crea le definizioni delle metriche
        metrics_definitions = {
            "accuracy": "Precisione nel suggerire film che corrispondono alle preferenze dell'utente basate sulla sua cronologia di valutazioni. Film con generi o attori simili a quelli che l'utente ha valutato positivamente in passato.",
            "diversity": "Varietà di generi, registi e stili di film nelle raccomandazioni. Film che coprono un'ampia gamma di categorie per espandere gli interessi dell'utente.",
            "novelty": "Film che l'utente probabilmente non conosce o non ha ancora scoperto, ma che potrebbero interessargli. Film meno mainstream o di nicchia che offrono nuove esperienze."
        }
        
        # Costruisci le catene di prompt per le metriche
        chains, parsers, _ = self.llm_service.build_metric_chains(metrics_definitions)
        
        # Dizionario per memorizzare i risultati
        metric_results = {}
        
        # Esegui i raccomandatori per ogni metrica
        for metric, chain in chains.items():
            print(f"Generazione raccomandazioni per metrica: {metric}")
            recommendations, raw_output = await self.llm_service.generate_recommendations(
                chain, 
                parsers[metric], 
                catalog, 
                user_profile_str
            )
            
            metric_results[metric] = {
                "recommendations": recommendations,
                "output": raw_output
            }
            
            print(f"Raccomandazioni generate per {metric}: {recommendations}")
        
        return metric_results
    
    async def evaluate_results(self, metric_results: Dict) -> Dict:
        """
        Valuta i risultati delle diverse metriche per generare raccomandazioni finali.
        
        Args:
            metric_results: Risultati delle metriche
            
        Returns:
            Dizionario con la valutazione finale
        """
        print("\n=== Valutazione dei risultati e generazione raccomandazioni finali ===\n")
        
        # Costruisci la catena di valutazione
        evaluator_chain, evaluator_parser = self.llm_service.build_evaluator_chain()
        
        # Valuta i risultati
        final_evaluation = await self.llm_service.evaluate_recommendations(
            evaluator_chain,
            evaluator_parser,
            metric_results
        )
        
        print(f"Raccomandazioni finali: {final_evaluation['final_recommendations']}")
        print(f"Giustificazione: {final_evaluation['justification']}")
        
        return final_evaluation
    
    async def generate_recommendations(self, user_id: int = 1, save_results: bool = True) -> Dict:
        """
        Genera raccomandazioni complete eseguendo tutte le fasi del processo.
        
        Args:
            user_id: ID dell'utente per cui generare raccomandazioni
            save_results: Se True, salva i risultati in un file JSON
            
        Returns:
            Dizionario con i risultati completi
        """
        # Carica i dataset se necessario
        if not self.datasets_loaded:
            self.load_datasets()
        
        # Esegui i raccomandatori per le diverse metriche
        metric_results = await self.run_metric_recommenders(user_id)
        
        # Valuta i risultati per generare raccomandazioni finali
        final_evaluation = await self.evaluate_results(metric_results)
        
        # Calcola le metriche
        metrics = calculate_metrics_for_recommendations(metric_results, final_evaluation)
        
        # Risultati completi
        results = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "metric_results": metric_results,
            "final_evaluation": final_evaluation,
            "metrics": metrics
        }
        
        # Salva i risultati
        if save_results:
            results_path = "recommendation_results.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            metric_results_path = "metric_recommendations.json"
            with open(metric_results_path, "w", encoding="utf-8") as f:
                json.dump(metric_results, f, ensure_ascii=False, indent=2)
            
            print(f"\nRisultati salvati in {results_path} e {metric_results_path}")
        
        return results 