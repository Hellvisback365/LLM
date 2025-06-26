"""
Script per convertire le liste classificate di raccomandazioni in triplette (user_id, item_id, probability).

Questo script:
1. Legge tutti i file checkpoint_batch_*.json
2. Estrae le raccomandazioni per ogni utente e metrica
3. Converte le posizioni in probabilità usando la formula: (N - i + 1) / N
4. Genera un file CSV con tutte le triplette
"""

import json
import csv
import os
import glob
from typing import Dict, List, Tuple
import sys

def calculate_probability(position: int, total_items: int = 20) -> float:
    """
    Calcola la probabilità basata sulla posizione nella lista classificata.
    
    Args:
        position: Posizione dell'item nella lista (1-based index)
        total_items: Numero totale di item nella lista (default: 20)
    
    Returns:
        Probabilità dell'item
    """
    return (total_items - position + 1) / total_items

def process_checkpoint_file(file_path: str) -> List[Tuple[str, str, float]]:
    """
    Processa un singolo file checkpoint e estrae le triplette.
    
    Args:
        file_path: Percorso del file checkpoint
    
    Returns:
        Lista di triplette (user_id, item_id, probability)
    """
    triplets = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        user_results = data.get('user_results', {})
        
        for user_id, user_data in user_results.items():
            # Processa ogni metrica per questo utente
            for metric_name, metric_data in user_data.items():
                if isinstance(metric_data, dict) and 'recommendations' in metric_data:
                    recommendations = metric_data['recommendations']
                    
                    # Genera triplette per ogni raccomandazione
                    for position, item_id in enumerate(recommendations, 1):
                        probability = calculate_probability(position)
                        # Creiamo un identificatore unico che include la metrica
                        unique_user_id = f"{user_id}_{metric_name}"
                        triplets.append((unique_user_id, str(item_id), probability))
        
        print(f"Processato {file_path}: {len(triplets)} triplette estratte")
        
    except Exception as e:
        print(f"Errore nel processare {file_path}: {e}")
    
    return triplets

def main():
    """Funzione principale per generare il file delle probabilità."""
    
    # Pattern per trovare tutti i file checkpoint
    checkpoint_pattern = "checkpoint_batch_*.json"
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"Nessun file trovato con pattern: {checkpoint_pattern}")
        print("Assicurati di essere nella directory corretta.")
        return
    
    # Ordina i file per numero di batch
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"Trovati {len(checkpoint_files)} file checkpoint")
    print("Iniziando il processamento...")
    
    all_triplets = []
    
    # Processa ogni file checkpoint
    for i, file_path in enumerate(checkpoint_files, 1):
        print(f"Processando file {i}/{len(checkpoint_files)}: {file_path}")
        triplets = process_checkpoint_file(file_path)
        all_triplets.extend(triplets)
    
    # Scrivi il file CSV di output
    output_file = "output_probabilities.csv"
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Scrivi l'header
            writer.writerow(['user_id', 'item_id', 'probability'])
            
            # Scrivi tutte le triplette
            writer.writerows(all_triplets)
        
        print(f"\nProcessamento completato!")
        print(f"File generato: {output_file}")
        print(f"Totale triplette generate: {len(all_triplets)}")
        
        # Statistiche aggiuntive
        unique_users = len(set(triplet[0] for triplet in all_triplets))
        unique_items = len(set(triplet[1] for triplet in all_triplets))
        
        print(f"Utenti unici (incluse metriche): {unique_users}")
        print(f"Item unici: {unique_items}")
        
        # Mostra alcuni esempi
        print(f"\nPrimi 10 esempi:")
        for i, (user_id, item_id, prob) in enumerate(all_triplets[:10]):
            print(f"  {user_id},{item_id},{prob:.3f}")
            
    except Exception as e:
        print(f"Errore nella scrittura del file CSV: {e}")

if __name__ == "__main__":
    main()
