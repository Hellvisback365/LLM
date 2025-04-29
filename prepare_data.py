import os
from agent import load_datasets, get_optimized_catalog
from src.recommender.utils.data_processor import process_dataset

print("\n=== Script di preparazione dati con filtro a 100 valutazioni ===\n")

# Forza il ricaricamento dei dataset
filtered_ratings, user_profiles, movies = load_datasets(force_reload=True)

print("\n=== Statistiche dataset filtrati ===")
print(f"Utenti: {len(user_profiles)}")
print(f"Film: {len(movies)}")
print(f"Valutazioni filtrate: {len(filtered_ratings)}")

# Ottieni il catalogo ottimizzato
catalog = get_optimized_catalog(limit=30)
catalog_sample = catalog[:500] + "..." if len(catalog) > 500 else catalog
print(f"\nCatalogo ottimizzato (campione): {catalog_sample}")

print("\n=== Preparazione dati completata ===")
print("I nuovi dati filtrati (soglia: 100 valutazioni) sono pronti per il sistema di raccomandazione.") 