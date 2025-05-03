# Multi-Metric Recommendation System

An advanced recommendation system based on **Large Language Models (LLMs)** that optimizes recommendations for **Precision@K** and **Coverage** metrics and combines them into a single optimal ranked list using a **Retrieval Augmented Generation (RAG)** approach.

## Features

- **Multi-Metric Optimization**
  - **Precision@K**: Recommends items the user is likely to rate positively.
  - **Coverage**: Maximizes genre diversity to reduce filter bubbles.
- **Hybrid Retrieval**: Combines vector search (FAISS) and BM25 for efficient candidate retrieval.
- **RAG-Augmented Generation**: Leverages LLMs with retrieval to generate high-quality, context-aware recommendations.
- **Automated Evaluation**: Computes quantitative metrics and generates detailed HTML/Markdown reports.
- **Configurable Experiments**: Easily run experiments with custom prompts and compare metric performance.

## Project Structure

```
project_root/
├── agent.py                      # Main script to run recommendation pipeline and generate reports
├── prepare_data.py               # Script to preprocess MovieLens dataset with threshold filtering
├── data/
│   ├── raw/                      # Raw MovieLens data files (.dat)
│   └── processed/                # Processed CSV files and RAG indices
├── src/
│   └── recommender/
│       ├── api/
│       │   └── llm_service.py    # LLM service wrapper (OpenAI/OpenRouter)
│       ├── core/
│       │   ├── recommender.py    # Core recommendation pipeline and agent definitions
│       │   └── metrics_calculator.py  # Functions to compute and aggregate metrics
│       └── utils/
│           ├── data_processor.py # Data loading, filtering, and user-profile creation
│           └── rag_utils.py      # RAG utilities: FAISS, BM25, prompt templates, embeddings
├── experiments/                  # Saved experiment outputs (JSON)
├── reports/                      # Generated HTML/Markdown experiment reports
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview and instructions
└── LICENSE                       # MIT License file
```

## Prerequisites

- **Python** 3.8 or higher
- **API Keys** (create a `.env` file in project root):
  ```ini
  OPENROUTER_API_KEY=your_openrouter_api_key
  OPENAI_API_KEY=your_openai_api_key  # optional, for embeddings
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tuonome/multi-metric-recommender.git
   cd multi-metric-recommender
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. Download the MovieLens 1M dataset from https://grouplens.org/datasets/movielens/1m/
2. Extract `movies.dat` and `ratings.dat` into the `data/raw/` folder.
3. Run the preparation script:
   ```bash
   python prepare_data.py
   ```

## Usage

Run the main recommendation pipeline with default settings:
```bash
python agent.py
```
This will execute the following steps:
1. Load and preprocess data.
2. Initialize RAG indices (FAISS & BM25).
3. Generate recommendations for each metric.
4. Combine, evaluate, and aggregate results.
5. Save outputs to `metric_recommendations.json` and `recommendation_results.json`.
6. Generate experiment reports in both HTML and Markdown formats (saved under `reports/`).

### Custom Experiments

You can define and run experiments with custom prompt variants:
```python
import asyncio
from src.recommender.core.recommender import RecommenderSystem

recommender = RecommenderSystem()

# Define custom prompts
custom_prompts = {
    "precision_at_k": "Ottimizza per film che l'utente valuterà positivamente...",
    "coverage": "Massimizza la varietà di generi nelle raccomandazioni..."
}

metrics, report_file = asyncio.run(
    recommender.generate_recommendations_with_custom_prompt(
        custom_prompts,
        experiment_name="my_experiment"
    )
)
print(f"Experiment report saved to {report_file}")
```

## Testing

Run unit and integration tests:
```bash
pytest
```

## Reporting

To generate detailed HTML/Markdown reports for past experiments:
```bash
python agent.py --analyze-experiments
```
Reports are saved to the `reports/` directory.

## Contributing

Contributions are welcome! Please open an issue to discuss major changes before submitting a pull request. Follow the [MIT License](LICENSE) guidelines.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details. 