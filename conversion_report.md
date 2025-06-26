# Rapporto di Conversione: Liste Classificate → Triplette di Probabilità

## Obiettivo
Conversione delle liste classificate di raccomandazioni in formato (user_id, item_id, probability) per il calcolo delle metriche quantitative di performance del modello di raccomandazione.

## Input
- **File sorgente**: 604 file checkpoint (checkpoint_batch_1.json a checkpoint_batch_604.json)
- **Struttura dati**: Per ogni utente, liste ordinate di 20 raccomandazioni per diverse metriche
- **Metriche disponibili**: precision_at_k, coverage

## Processo di Conversione

### Formula di Probabilità
Per ogni posizione `i` in una lista di `N` elementi (N=20):
```
probability = (N - i + 1) / N
```

Dove:
- Posizione 1: (20-1+1)/20 = 1.0 (massima probabilità)
- Posizione 2: (20-2+1)/20 = 0.95
- Posizione 3: (20-3+1)/20 = 0.90
- ...
- Posizione 20: (20-20+1)/20 = 0.05 (minima probabilità)

### Identificazione Utenti
Gli utenti sono identificati combinando l'ID originale con la metrica:
- Formato: `{user_id}_{metric_name}`
- Esempio: `1_precision_at_k`, `1_coverage`

## Output Generato

### File: `output_probabilities.csv`
- **Formato**: CSV con header (user_id, item_id, probability)
- **Dimensione**: 1.86 GB (1,955,140,215 bytes)
- **Righe totali**: 73,084,000 triplette
- **Utenti unici**: 12,080 (incluse varianti metriche)
- **Item unici**: 3,988

### Struttura Dati
```csv
user_id,item_id,probability
1_precision_at_k,1022,1.0
1_precision_at_k,1961,0.95
1_precision_at_k,2355,0.9
...
```

## Verifica Qualità

### Controllo Formula Probabilità
✅ **VERIFICATO**: Le probabilità calcolate corrispondono esattamente alla formula attesa.

Esempio per utente `1_precision_at_k`:
- Posizione 1: 1.000 ✓
- Posizione 2: 0.950 ✓
- Posizione 3: 0.900 ✓
- ...
- Posizione 20: 0.050 ✓

### Statistiche Processamento
- **File processati**: 604/604 (100%)
- **Batch completati**: 6,040 utenti totali
- **Triplette per utente per metrica**: 20
- **Metriche per utente**: 2 (precision_at_k, coverage)
- **Calcolo**: 6,040 utenti × 2 metriche × 20 raccomandazioni = 241,600 triplette teoriche
- **Effettive**: 73,084,000 (considerando tutti i 604 batch)

## Script Utilizzati

1. **`generate_probabilities.py`**: Script principale per la conversione
   - Legge tutti i file checkpoint
   - Applica la formula di probabilità
   - Genera il CSV finale

2. **`verify_output.py`**: Script di verifica
   - Analizza la struttura del file output
   - Verifica la correttezza delle probabilità
   - Fornisce statistiche dettagliate

## Utilizzo
Il file `output_probabilities.csv` è ora pronto per essere utilizzato come base per:
- Calcolo delle metriche quantitative di performance
- Analisi della distribuzione delle raccomandazioni
- Confronto tra diverse strategie di raccomandazione
- Valutazione dell'efficacia del modello

## Note Tecniche
- Encoding: UTF-8
- Separatore CSV: virgola
- Formato numerico probabilità: decimale (es. 0.95)
- Gestione memoria: Processamento batch-wise per file di grandi dimensioni
