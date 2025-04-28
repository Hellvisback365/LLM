# Sistema di Raccomandazione Multi-Metrica

Questo progetto implementa un sistema di raccomandazione che utilizza LangChain e modelli LLM per generare raccomandazioni ottimizzate per diverse metriche (accuratezza, diversità, novità) e poi combinarle in un'unica lista ottimale.

## Struttura del Progetto

```
/project_root
├── agent.py              # Script principale per il sistema di raccomandazione
├── data/                 # Directory per i dataset
│   ├── raw/              # Dataset grezzi in formato .dat (movies.dat, ratings.dat)
│   └── processed/        # Dataset processati e catalogo ottimizzato
├── utils/
│   └── data_processor.py      # Funzioni per processare i dataset
│   └── rag_utils.py           # Funzioni per RAG e metriche
├── config/
│   └── settings.py            # Impostazioni globali (opzionale)
└── .env                       # File per le variabili d'ambiente (API keys)
```

## Prerequisiti

- Python 3.8+
- API key per OpenRouter (utilizzata per accedere a Mistral 7B)
- Dataset MovieLens (files: movies.dat, ratings.dat nella cartella data/raw)

## Installazione

1. Clona il repository
2. Crea un ambiente virtuale:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```
3. Installa le dipendenze:
   ```
   pip install -r requirements.txt
   ```
4. Crea un file `.env` nella root del progetto con la tua API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Esecuzione

Per eseguire il sistema di raccomandazione:

```
python agent.py
```

Il sistema seguirà questi passaggi:
1. Carica e processa i dataset
2. Crea un catalogo ottimizzato usando RAG
3. Genera raccomandazioni per ogni metrica (accuratezza, diversità, novità)
4. Combina le raccomandazioni in un'unica lista ottimale
5. Salva i risultati nei file `metric_recommendations.json` e `recommendation_results.json`

## Metriche supportate

- **Accuratezza**: Ottimizza per film popolari e con alta probabilità di piacere all'utente
- **Diversità**: Ottimizza per film di generi diversi, massimizzando la copertura di categorie
- **Novità**: Ottimizza per film originali o meno mainstream

## Note

- Il sistema utilizza un approccio RAG (Retrieval Augmented Generation) per ottimizzare il catalogo di film
- Le metriche vengono bilanciate nell'output finale per offrire raccomandazioni ottimali
- I risultati intermedi e finali vengono salvati in formato JSON per successive analisi 