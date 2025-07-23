# LLM-Based Multi-Metric Movie Recommendation System

Un sistema di raccomandazione cinematografica avanzato che utilizza Large Language Models (LLM) per generare raccomandazioni multi-metriche intelligenti, bilanciando precision@k e coverage attraverso un'architettura modulare basata su LangGraph.

## 🎯 Panoramica del Progetto

Questo sistema implementa una pipeline di raccomandazione che:
- Utilizza **Meta-Llama 3.2 3B** via OpenRouter per generare raccomandazioni intelligenti
- Combina **precision@k** (accuratezza) e **coverage** (diversità) utilizzando strategie LLM
- Implementa un sistema **RAG (Retrieval-Augmented Generation)** con FAISS e BM25
- Supporta **esperimenti con prompt personalizzati** per ottimizzare diverse metriche
- Genera **report automatici** in HTML e Markdown per l'analisi dei risultati

### 🔍 Agenti di Aggregazione

Il sistema implementa **due distinti agenti aggregatori**:

1. **Aggregatore LangGraph** (`src/recommender/core/recommender.py` + `src/recommender/core/graph_nodes.py`)
   - Utilizzato per i test iniziali e sviluppo
   - Effettua aggregazione su **gruppi di utenti** 
   - Basato su workflow LangGraph per orchestrazione complessa

2. **Aggregatore LLM Pulito** (`replace_aggregation_with_llm.py`)
   - **Utilizzato per i risultati finali** del progetto
   - Effettua aggregazione **per ogni singolo profilo utente**
   - Implementazione ottimizzata con processamento parallelo e cache
   - Genera file `llm_aggregated_recommendations.json` con risultati finali

## 📁 Struttura del Progetto

```
Tesi_py/
├── 📁 src/                           # Codice sorgente principale
│   ├── 📁 recommender/
│   │   ├── 📁 core/                  # Componenti core del sistema
│   │   │   ├── recommender.py        # Sistema principale + aggregatore LangGraph
│   │   │   ├── graph_nodes.py        # Nodi LangGraph per workflow
│   │   │   ├── metrics_utils.py      # Calcolo metriche di valutazione
│   │   │   ├── prompt_manager.py     # Gestione template prompts
│   │   │   ├── schemas.py            # Schemi Pydantic per validazione
│   │   │   └── experiment_manager.py # Gestione esperimenti automatizzati
│   │   └── 📁 utils/                 # Utilità di supporto
│   │       ├── data_processor.py     # Caricamento e preprocessing dati
│   │       ├── rag_utils.py          # Sistema RAG (FAISS + BM25)
│   │       └── data_utils.py         # Utilità conversione dati
│   └── 📁 reporting/                 # Sistema di reporting
│       └── experiment_reporter.py    # Generazione report HTML/MD
├── 📁 data/                          # Dataset e indici
│   ├── 📁 raw/                       # Dati MovieLens originali
│   ├── 📁 processed/                 # Dati preprocessati (CSV)
│   ├── 📁 rag/                       # Indici FAISS e BM25
│   └── 📁 temp/                      # File temporanei
├── 📁 experiments/                   # Risultati esperimenti salvati
├── 📁 reports/                       # Report generati (HTML/MD)
├── 📁 checkpoint_batch_*.json        # Checkpoint processamento (604 file)
├── agent.py                          # Script principale per esperimenti
├── replace_aggregation_with_llm.py   # 🎯 Aggregatore finale (risultati usati)
├── analyzer.py                       # Script analisi risultati
├── requirements.txt                  # Dipendenze Python
├── langgraph_requirements.txt        # Dipendenze specifiche LangGraph
└── llm_aggregated_recommendations.json # 🎯 Risultati finali aggregati
```

## 🛠️ Requisiti e Dipendenze

### Prerequisiti
- **Python** 3.8 o superiore
- **Chiavi API** (crea un file `.env` nella root del progetto):
  ```ini
  OPENROUTER_API_KEY=your_openrouter_api_key
  OPENAI_API_KEY=your_openai_api_key  # opzionale, per embeddings
  ```

### Dipendenze Principali
```
langchain>=0.1.0              # Framework LLM principale
langchain-openai>=0.0.5       # Integrazione OpenAI/OpenRouter
langchain-community>=0.0.17   # Componenti community (FAISS)
langgraph>=0.0.19            # Orchestrazione workflow
pandas>=2.2.0                # Manipolazione dati
numpy>=1.26.0                # Calcoli numerici
faiss-cpu>=1.7.4             # Vector database per RAG
rank-bm25>=0.2.2             # Algoritmo BM25 per retrieval
sentence-transformers>=2.2.2  # Cross-encoder per ranking
tenacity>=8.2.3              # Retry logic per API calls
tqdm>=4.66.1                 # Progress bars
python-dotenv>=1.0.0         # Gestione variabili ambiente
```

## 🚀 Installazione

1. **Clona il repository:**
   ```bash
   git clone https://github.com/Hellvisback365/LLM.git
   cd LLM
   ```

2. **Crea e attiva un ambiente virtuale:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Installa le dipendenze:**
   ```bash
   pip install -r requirements.txt
   pip install -r langgraph_requirements.txt
   ```

4. **Configura le variabili d'ambiente:**
   ```bash
   # Crea file .env
   echo "OPENROUTER_API_KEY=your_key_here" > .env
   echo "OPENAI_API_KEY=your_openai_key" >> .env
   ```

## 📖 Utilizzo

### Generazione Raccomandazioni Standard

**Script principale per esperimenti iniziali:**
```bash
python agent.py
```

**Aggregatore finale (per risultati definitivi):**
```bash
python replace_aggregation_with_llm.py
```

Questo script:
1. Processa tutti i checkpoint esistenti
2. Genera aggregazioni LLM per ogni utente individualmente
3. Salva i risultati in `llm_aggregated_recommendations.json`
4. Supporta modalità resume automatica

### Esperimenti Personalizzati

```python
import asyncio
from src.recommender.core.recommender import RecommenderSystem

# Inizializza il sistema
recommender = RecommenderSystem()
recommender.initialize_system()

# Definisci prompt personalizzati
custom_prompts = {
    "precision_at_k": """Ottimizza per film che l'utente valuterà 4-5 stelle.
    Analizza generi preferiti, attori e registi del profilo utente...""",
    "coverage": """Massimizza diversità di generi mantenendo rilevanza.
    Bilancia esplorazione con preferenze personali..."""
}

# Esegui esperimento
metrics, report_file = await recommender.generate_recommendations_with_custom_prompt(
    custom_prompts,
    experiment_name="mio_esperimento"
)
print(f"Report salvato in: {report_file}")
```

### Analisi e Reporting

**Genera report automatici per tutti gli esperimenti:**
```bash
python agent.py --analyze-experiments
```

**Analisi risultati specifici:**
```python
from src.reporting.experiment_reporter import ExperimentReporter

reporter = ExperimentReporter(experiments_dir="experiments")
summary = reporter.run_comprehensive_analysis(output_dir="reports")
print(f"Esperimenti analizzati: {summary['total_experiments']}")
```

## 🏗️ Architettura del Sistema

### Core Components

1. **RecommenderSystem** (`src/recommender/core/recommender.py`)
   - Orchestratore principale basato su LangGraph
   - Gestisce workflow parallelo per precision@k e coverage
   - Integra sistema RAG per retrieval intelligente

2. **LLMAggregatorReplacer** (`replace_aggregation_with_llm.py`)
   - Aggregatore finale per risultati production
   - Processamento parallelo ottimizzato con semafori
   - Cache catalogo per performance elevate

3. **MovieRAG** (`src/recommender/utils/rag_utils.py`)
   - Sistema di retrieval con FAISS (vector search) + BM25 (keyword search)
   - Cross-encoder per re-ranking dei risultati
   - Embeddings OpenAI per rappresentazione semantica

4. **MetricsCalculator** (`src/recommender/core/metrics_utils.py`)
   - Calcolo precision@k, coverage, diversità temporale
   - Metriche di genre entropy e item coverage
   - Validazione statistica dei risultati


## 🧪 Metriche di Valutazione

Il sistema calcola automaticamente:

- **Precision@K**: Percentuale di raccomandazioni relevant nei top-K
- **Coverage**: Proporzione del catalogo coperta dalle raccomandazioni
- **Genre Coverage**: Diversità di generi nelle raccomandazioni
- **Temporal Dispersion**: Distribuzione temporale dei film raccomandati
- **Genre Entropy**: Entropia nella distribuzione dei generi

## 📊 Comandi Disponibili

```bash
# Esecuzione principale
python agent.py                              # Esperimenti con aggregatore LangGraph
python replace_aggregation_with_llm.py       # Aggregazione finale ottimizzata

# Configurazione ambiente
export BATCH_SIZE=20                         # Dimensione batch processing
export RUN_EXPERIMENTS=true                  # Abilita esperimenti automatici

# Testing e validazione
python -m pytest tests/                      # Esegui test unitari
python analyzer.py                          # Analisi risultati esistenti

# Generazione report
python -c "from src.reporting.experiment_reporter import ExperimentReporter; 
           ExperimentReporter().run_comprehensive_analysis()"
```

## 🔧 Configurazione Avanzata

### Parametri LLM
```python
# In src/recommender/core/recommender.py
COMMON_LLM_PARAMS = {
    "temperature": 0.2,        # Bilanciamento creatività/consistenza
    "max_tokens": 4000,        # Spazio per risposte complete
    "model": "meta-llama/llama-3.2-3b-instruct"
}
```

### Ottimizzazioni Performance
```python
# In replace_aggregation_with_llm.py
batch_size = 50               # Utenti processati in parallelo
semaphore = asyncio.Semaphore(50)  # Limite chiamate LLM concorrenti
checkpoint_batch_size = 10    # Checkpoint processati insieme
```

## 🤝 Contribuzione

1. **Fork** del repository
2. **Crea** un branch per la feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** delle modifiche (`git commit -m 'Add AmazingFeature'`)
4. **Push** al branch (`git push origin feature/AmazingFeature`)
5. **Apri** una Pull Request

### Guidelines
- Segui PEP 8 per lo stile Python
- Aggiungi test per nuove funzionalità
- Documenta API pubbliche con docstring
- Usa type hints per migliorare la leggibilità

## 📄 Licenza

Distribuito sotto licenza MIT. Vedi [LICENSE](LICENSE) per dettagli.

## 📞 Contatti

- **Repository**: [https://github.com/Hellvisback365/LLM](https://github.com/Hellvisback365/LLM)


## 🎯 Risultati e Performance

Il sistema ha processato con successo:
- **6040 utenti** del dataset MovieLens
- **604 checkpoint** di processamento
- **Tasso di successo LLM**: >95% per aggregazioni finali
- **Tempo medio**: ~0.5 secondi per utente con ottimizzazioni

I risultati finali sono disponibili in `llm_aggregated_recommendations.json` generato dall'aggregatore ottimizzato.