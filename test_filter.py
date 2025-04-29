from src.recommender.utils.data_processor import load_ratings, filter_users_by_min_ratings, create_user_profiles
import os
import pandas as pd

print("=== Test di filtraggio con soglia di 100 valutazioni ===\n")

# Carica i dati delle valutazioni
print("Caricamento dei dati di valutazione...")
ratings = load_ratings()
print(f"Dati caricati: {len(ratings)} valutazioni di {ratings['user_id'].nunique()} utenti\n")

# Applica il filtro (min_ratings=100 è ora il valore predefinito)
print("Applicazione del filtro...")
filtered_ratings = filter_users_by_min_ratings(ratings)
print(f"Filtraggio completato: {filtered_ratings['user_id'].nunique()} utenti rimasti\n")

# Crea profili utente dai dati filtrati
print("Creazione profili utente...")
user_profiles = create_user_profiles(filtered_ratings)
print(f"Profili creati: {len(user_profiles)} utenti\n")

# Salva i dati filtrati in file temporanei per ispezione
out_dir = "data/temp"
os.makedirs(out_dir, exist_ok=True)

print("Salvataggio dei dati filtrati...")
filtered_ratings.to_csv(os.path.join(out_dir, "filtered_ratings_100.csv"), index=False)
user_profiles.to_csv(os.path.join(out_dir, "user_profiles_100.csv"))

print("\nAnalisi delle valutazioni per utente nei dati filtrati:")
user_rating_counts = filtered_ratings['user_id'].value_counts()
print(f"Minimo valutazioni: {user_rating_counts.min()}")
print(f"Massimo valutazioni: {user_rating_counts.max()}")
print(f"Media valutazioni: {user_rating_counts.mean():.2f}")

print("\nDistribuzione delle valutazioni:")
ranges = [(100, 200), (201, 500), (501, 1000), (1001, float('inf'))]
for lower, upper in ranges:
    if upper == float('inf'):
        count = sum(1 for c in user_rating_counts if c >= lower)
        percent = count / len(user_rating_counts) * 100
        print(f"{lower}+ valutazioni: {count} utenti ({percent:.2f}%)")
    else:
        count = sum(1 for c in user_rating_counts if lower <= c <= upper)
        percent = count / len(user_rating_counts) * 100
        print(f"{lower}-{upper} valutazioni: {count} utenti ({percent:.2f}%)")

print("\nTest completato. Dati filtrati salvati in data/temp/") 