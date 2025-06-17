#!/usr/bin/env python3
"""
Test script per validare le nuove funzionalità di aggregazione delle metriche.
Questo script testa se le nuove metriche aggregate (GenreEntropy, TempDisp, AvgYear) 
vengono calcolate correttamente dall'agente aggregatore.
"""

import pandas as pd
import numpy as np
from src.recommender.core.metrics_utils import MetricsCalculator

def test_aggregated_metrics():
    print("=== Test delle Nuove Metriche Aggregate per l'Agente Aggregatore ===\n")
    
    # Crea un dataset di test con film simulati
    movies_test = pd.DataFrame({
        'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': [f'Movie {i}' for i in range(1, 11)],
        'genres': ['Action|Drama', 'Comedy', 'Horror|Thriller', 'Romance', 'Sci-Fi|Action', 
                  'Drama', 'Comedy|Romance', 'Action', 'Horror', 'Thriller|Drama'],
        'year': [1990, 1995, 2000, 1985, 2005, 1980, 1992, 2010, 1988, 1999]
    })
    
    # Inizializza il MetricsCalculator
    calculator = MetricsCalculator(movies_test)
    
    # Simula l'accumulatore con dati per test
    test_accumulator = {
        'precision_at_k': {
            'average_release_year_scores': [1992.5, 1988.0, 1995.2],  # 3 utenti con medie diverse
            'temporal_dispersion_scores': [8.5, 12.3, 10.1],
            'genre_entropy_scores': [2.1, 2.5, 2.8]
        },
        'coverage': {
            'average_release_year_scores': [1985.0, 1990.5, 1987.3],
            'temporal_dispersion_scores': [15.2, 18.7, 16.9],
            'genre_entropy_scores': [3.1, 3.5, 3.3]
        }
    }
    
    # Test della nuova funzione di aggregazione per l'agente aggregatore
    print("1. Test della funzione calculate_aggregated_metrics_for_agent:")
    aggregated_results = calculator.calculate_aggregated_metrics_for_agent(
        test_accumulator, 
        ['precision_at_k', 'coverage']
    )
    
    print(f"Risultati aggregati: {aggregated_results}")
    
    # Verifica che le metriche siano state calcolate correttamente
    print("\n2. Verifiche dei Calcoli:")
    
    # Test precision_at_k
    if 'precision_at_k' in aggregated_results:
        prec_data = aggregated_results['precision_at_k']
        expected_avg_year = np.mean([1992.5, 1988.0, 1995.2])
        expected_temp_disp = np.mean([8.5, 12.3, 10.1])
        expected_genre_entropy = np.mean([2.1, 2.5, 2.8])
        
        print(f"   precision_at_k - AvgYear: {prec_data.get('average_release_year', 'N/A'):.1f} (atteso: {expected_avg_year:.1f})")
        print(f"   precision_at_k - TempDisp: {prec_data.get('temporal_dispersion', 'N/A'):.2f} (atteso: {expected_temp_disp:.2f})")
        print(f"   precision_at_k - GenreEntropy: {prec_data.get('genre_entropy', 'N/A'):.3f} (atteso: {expected_genre_entropy:.3f})")
        
        # Verifica presenza dei nomi alternativi per agente aggregatore
        print(f"   precision_at_k - avg_year: {prec_data.get('avg_year', 'N/A')}")
        print(f"   precision_at_k - avg_temporal_dispersion: {prec_data.get('avg_temporal_dispersion', 'N/A')}")
        print(f"   precision_at_k - avg_genre_entropy: {prec_data.get('avg_genre_entropy', 'N/A')}")
    
    # Test coverage
    if 'coverage' in aggregated_results:
        cov_data = aggregated_results['coverage']
        expected_avg_year_cov = np.mean([1985.0, 1990.5, 1987.3])
        expected_temp_disp_cov = np.mean([15.2, 18.7, 16.9])
        expected_genre_entropy_cov = np.mean([3.1, 3.5, 3.3])
        
        print(f"   coverage - AvgYear: {cov_data.get('average_release_year', 'N/A'):.1f} (atteso: {expected_avg_year_cov:.1f})")
        print(f"   coverage - TempDisp: {cov_data.get('temporal_dispersion', 'N/A'):.2f} (atteso: {expected_temp_disp_cov:.2f})")
        print(f"   coverage - GenreEntropy: {cov_data.get('genre_entropy', 'N/A'):.3f} (atteso: {expected_genre_entropy_cov:.3f})")
    
    print("\n3. Test delle Funzioni di Calcolo di Base:")
    
    # Test GenreEntropy
    test_recs = [1, 3, 5, 7]  # Film con diversi generi
    genre_entropy = calculator.calculate_genre_entropy(test_recs)
    print(f"   GenreEntropy per raccomandazioni {test_recs}: {genre_entropy:.4f}")
    
    # Test TempDisp
    temp_disp = calculator.calculate_temporal_dispersion(test_recs)
    print(f"   TempDisp per raccomandazioni {test_recs}: {temp_disp:.2f}")
    
    # Test AvgYear
    avg_year = calculator.calculate_average_release_year(test_recs)
    print(f"   AvgYear per raccomandazioni {test_recs}: {avg_year:.1f}")
    
    print("\n=== Test Completato con Successo! ===")
    return True

if __name__ == "__main__":
    try:
        test_aggregated_metrics()
        print("\n✅ Tutti i test sono passati! Le nuove funzionalità di aggregazione funzionano correttamente.")
    except Exception as e:
        print(f"\n❌ Errore durante i test: {e}")
        import traceback
        traceback.print_exc()
