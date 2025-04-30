#!/usr/bin/env python3
"""
Test di verifica della fusione dei cataloghi in rag_utils.py
Questo script verifica che la fusione tra i cataloghi di precision@k e coverage
funzioni correttamente e produca un catalogo bilanciato.
"""

import os
import json
import pandas as pd
import random
from pprint import pprint
from collections import Counter

# Importiamo la classe MovieRAG e le funzioni di metriche
from src.recommender.utils.rag_utils import MovieRAG, calculate_precision_at_k, calculate_coverage

def print_catalog_stats(catalog, name):
    """Stampa statistiche su un catalogo"""
    print(f"\n=== Statistiche catalogo {name} ===")
    print(f"Numero di film: {len(catalog)}")
    
    # Analisi dei generi
    genre_counter = Counter()
    for movie in catalog:
        genres = str(movie.get('genres', '')).split('|')
        for genre in genres:
            if genre.strip():
                genre_counter[genre.strip()] += 1
    
    print("\nDistribuzione generi:")
    for genre, count in genre_counter.most_common():
        print(f"  {genre}: {count} film ({count/len(catalog)*100:.1f}%)")
    
    # IDs dei film per verifica overlap
    movie_ids = [movie['movie_id'] for movie in catalog]
    return movie_ids

def check_catalog_overlap(ids1, ids2):
    """Verifica la sovrapposizione tra due cataloghi"""
    set1 = set(ids1)
    set2 = set(ids2)
    
    overlap = set1.intersection(set2)
    
    print(f"\n=== Sovrapposizione tra cataloghi ===")
    print(f"Catalogo 1: {len(ids1)} film")
    print(f"Catalogo 2: {len(ids2)} film")
    print(f"Film in comune: {len(overlap)} ({len(overlap)/len(ids1)*100:.1f}% del catalogo 1)")
    
    return overlap

def check_merged_catalog(merged, ids1, ids2):
    """Verifica la correttezza del catalogo unito"""
    merged_ids = [movie['movie_id'] for movie in merged]
    
    precision_only = set(ids1) - set(ids2)
    coverage_only = set(ids2) - set(ids1)
    
    precision_included = sum(1 for id in precision_only if id in merged_ids)
    coverage_included = sum(1 for id in coverage_only if id in merged_ids)
    
    print(f"\n=== Analisi catalogo unito ===")
    print(f"Catalogo unito: {len(merged)} film")
    print(f"Film unici da precision inclusi: {precision_included} di {len(precision_only)}")
    print(f"Film unici da coverage inclusi: {coverage_included} di {len(coverage_only)}")
    
    # Verifica alternanza
    sequence_match = True
    seen_ids = set()
    i, j = 0, 0
    
    # Test primi 10 elementi per vedere se c'è alternanza
    for k in range(min(10, len(merged))):
        if i < len(ids1) and ids1[i] not in seen_ids:
            if merged_ids[k] == ids1[i]:
                seen_ids.add(ids1[i])
                i += 1
                continue
                
        if j < len(ids2) and ids2[j] not in seen_ids:
            if merged_ids[k] == ids2[j]:
                seen_ids.add(ids2[j])
                j += 1
                continue
                
        sequence_match = False
        break
    
    print(f"Alternanza dei cataloghi rispettata: {sequence_match}")
    
    # Verifica della distribuzione dei generi
    print_catalog_stats(merged, "unito")

def main():
    """Funzione principale"""
    print("=== Test fusione cataloghi ===\n")
    
    # Carica i film dal dataset
    data_dir = os.path.join('data', 'processed')
    movies_path = os.path.join(data_dir, 'movies.csv')
    
    if not os.path.exists(movies_path):
        print(f"File {movies_path} non trovato. Assicurati di aver generato il dataset processato.")
        return
    
    # Carica i film
    movies_df = pd.read_csv(movies_path)
    movies_list = movies_df.to_dict('records')
    
    print(f"Caricati {len(movies_list)} film dal dataset")
    
    # Inizializza MovieRAG
    rag = MovieRAG()
    rag.initialize_data(movies_list)
    
    # Genera i cataloghi individuali
    precision_catalog = rag.generate_metrics_optimized_catalog(movies_list, 'precision_at_k')
    coverage_catalog = rag.generate_metrics_optimized_catalog(movies_list, 'coverage')
    
    # Genera il catalogo unito
    merged_catalog = rag.merge_catalogs(precision_catalog, coverage_catalog)
    
    # Analizza i cataloghi
    precision_ids = print_catalog_stats(precision_catalog, "precision@k")
    coverage_ids = print_catalog_stats(coverage_catalog, "coverage")
    
    # Verifica overlap
    check_catalog_overlap(precision_ids, coverage_ids)
    
    # Verifica fusione
    check_merged_catalog(merged_catalog, precision_ids, coverage_ids)
    
    # Salva i risultati
    os.makedirs('results', exist_ok=True)
    
    with open('results/precision_catalog.json', 'w', encoding='utf-8') as f:
        json.dump(precision_catalog, f, ensure_ascii=False, indent=2)
        
    with open('results/coverage_catalog.json', 'w', encoding='utf-8') as f:
        json.dump(coverage_catalog, f, ensure_ascii=False, indent=2)
        
    with open('results/merged_catalog.json', 'w', encoding='utf-8') as f:
        json.dump(merged_catalog, f, ensure_ascii=False, indent=2)
    
    print("\nCataloghi salvati nella directory 'results'")
    
    # Test del catalogo JSON finale
    final_catalog = rag.get_optimized_catalog_for_llm(movies_list, limit=100)
    with open('results/final_optimized_catalog.json', 'w', encoding='utf-8') as f:
        f.write(final_catalog)
    
    print(f"Catalogo ottimizzato finale salvato in 'results/final_optimized_catalog.json'")

if __name__ == "__main__":
    main() 