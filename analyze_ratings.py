import os
from collections import defaultdict
import pandas as pd

# Percorso del file ratings.dat
ratings_file = os.path.join("data", "raw", "ratings.dat")
filtered_ratings_file = os.path.join("data", "processed", "filtered_ratings.csv")

# Dizionario per contare le valutazioni per utente
user_ratings_count = defaultdict(int)

# Leggi il file e conta le valutazioni per ciascun utente
print(f"Analisi del file: {ratings_file}")
print("Conteggio valutazioni per utente...")

try:
    with open(ratings_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 3:
                user_id = int(parts[0])
                user_ratings_count[user_id] += 1
    
    # Trova utenti con meno di 10 valutazioni
    users_with_few_ratings = {user_id: count for user_id, count in user_ratings_count.items() if count < 10}
    
    print(f"\nTotale utenti nel dataset: {len(user_ratings_count)}")
    print(f"Utenti con meno di 10 valutazioni: {len(users_with_few_ratings)}")
    
    # Calcola statistiche sulle valutazioni per utente
    counts = list(user_ratings_count.values())
    
    if counts:
        print(f"\nStatistiche delle valutazioni per utente:")
        print(f"Minimo: {min(counts)} valutazioni")
        print(f"Massimo: {max(counts)} valutazioni")
        print(f"Media: {sum(counts)/len(counts):.2f} valutazioni")
        
        # Mostra la distribuzione delle valutazioni in intervalli
        print("\nDistribuzione del numero di valutazioni per utente:")
        
        # Definiamo degli intervalli per un'analisi più dettagliata
        ranges = [(10, 20), (21, 50), (51, 100), (101, 200), (201, 500), (501, 1000), (1001, float('inf'))]
        
        for lower, upper in ranges:
            if upper == float('inf'):
                count_in_range = sum(1 for c in counts if c >= lower)
                print(f"{lower}+ valutazioni: {count_in_range} utenti ({count_in_range/len(user_ratings_count)*100:.2f}%)")
            else:
                count_in_range = sum(1 for c in counts if lower <= c <= upper)
                print(f"{lower}-{upper} valutazioni: {count_in_range} utenti ({count_in_range/len(user_ratings_count)*100:.2f}%)")
    
    # Verifica con il file filtrato se esiste
    if os.path.exists(filtered_ratings_file):
        try:
            # Carica il file filtrato
            filtered_ratings = pd.read_csv(filtered_ratings_file)
            
            # Conta gli utenti unici nel file filtrato
            filtered_user_count = filtered_ratings['user_id'].nunique()
            
            print(f"\nVerifica con il file filtrato:")
            print(f"Utenti nel file originale: {len(user_ratings_count)}")
            print(f"Utenti nel file filtrato: {filtered_user_count}")
            
            if filtered_user_count == len(user_ratings_count):
                print("✓ Il filtro funziona correttamente: tutti gli utenti hanno almeno 10 valutazioni.")
            else:
                print("✗ Il filtro non funziona come previsto!")
                print(f"Differenza: {len(user_ratings_count) - filtered_user_count} utenti mancanti nel file filtrato.")
        except Exception as e:
            print(f"Errore nell'analisi del file filtrato: {e}")
    else:
        print(f"\nIl file filtrato non esiste: {filtered_ratings_file}")

except Exception as e:
    print(f"Errore durante l'analisi del file: {e}") 