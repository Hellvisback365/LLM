"""
Script per verificare la struttura e le statistiche del file output_probabilities.csv
"""

import csv
import os

def analyze_output_file():
    file_path = "output_probabilities.csv"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} non trovato!")
        return
    
    # Informazioni sul file
    file_size = os.path.getsize(file_path)
    print(f"Dimensione file: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    # Analizza il contenuto
    line_count = 0
    unique_users = set()
    unique_items = set()
    
    print("Leggendo il file...")
    
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        # Leggi l'header
        header = next(reader)
        print(f"Header: {header}")
        
        # Analizza le prime righe
        print("\nPrime 10 righe di dati:")
        for i, row in enumerate(reader):
            if i < 10:
                print(f"  {','.join(row)}")
            
            line_count += 1
            if len(row) >= 3:
                unique_users.add(row[0])
                unique_items.add(row[1])
            
            # Per file molto grandi, campiona ogni 1000a riga dopo le prime 1000
            if line_count > 1000 and line_count % 1000 != 0:
                continue
    
    print(f"\nStatistiche:")
    print(f"  - Righe totali (escluso header): {line_count:,}")
    print(f"  - Utenti unici: {len(unique_users):,}")
    print(f"  - Item unici: {len(unique_items):,}")
    
    # Verifica la formula delle probabilità
    print(f"\nVerifica formula probabilità:")
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        
        # Prendi le prime 20 righe dello stesso utente
        first_user = None
        probabilities = []
        
        for row in reader:
            if len(row) >= 3:
                user_id = row[0]
                probability = float(row[2])
                
                if first_user is None:
                    first_user = user_id
                
                if user_id == first_user:
                    probabilities.append(probability)
                    if len(probabilities) >= 20:
                        break
        
        print(f"  - Prime 20 probabilità per utente {first_user}:")
        for i, prob in enumerate(probabilities, 1):
            expected = (20 - i + 1) / 20
            print(f"    Posizione {i}: {prob:.3f} (atteso: {expected:.3f})")

if __name__ == "__main__":
    analyze_output_file()
