import asyncio
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

# Load environment
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in environment")

# Common LLM parameters
COMMON_LLM_PARAMS = {
    "openai_api_base": "https://openrouter.ai/api/v1",
    "openai_api_key": OPENROUTER_API_KEY,
    "temperature": 0.7,
    "max_tokens": 512,
}

# Model configurations
MODEL_CONFIGS = [
    {"name": "llama_4_maverick", "model_id": "meta-llama/llama-4-maverick:free"},
    {"name": "llama_4_maverick",      "model_id": "meta-llama/llama-4-maverick:free"},
    {"name": "llama_4_maverick",      "model_id": "meta-llama/llama-4-maverick:free"},
]

# Toy catalog of food products
TOY_CATALOG = [
    {"id":1, "name":"Margherita Pizza"},
    {"id":2, "name":"Sushi Roll"},
    {"id":3, "name":"Falafel Wrap"},
    {"id":4, "name":"Veggie Burger"},
    {"id":5, "name":"Pad Thai"},
    {"id":6, "name":"Tacos al Pastor"},
    {"id":7, "name":"Caesar Salad"},
    {"id":8, "name":"Ramen Bowl"},
]

# Prompt variants targeting different metrics
PROMPT_VARIANTS = {
    "accuracy":    "Suggerisci i top 3 prodotti food più rilevanti per l'utente, massimizzando l'accuratezza.",
    "novelty":     "Suggerisci i top 3 prodotti food più innovativi e nuovi per l'utente, massimizzando la novità.",
    "serendipity": "Suggerisci i top 3 prodotti food più sorprendenti ma apprezzabili, massimizzando la serendipità.",
    "coverage":    "Suggerisci una lista di prodotti food che copra il più possibile il catalogo, massimizzando la copertura.",
}

# Initialize LLM chains for each model using pipe syntax
def build_model_chains():
    chains = {}
    for cfg in MODEL_CONFIGS:
        llm = ChatOpenAI(model=cfg['model_id'], **COMMON_LLM_PARAMS)
        for metric, prompt_text in PROMPT_VARIANTS.items():
            key = f"{cfg['name']}_{metric}"
            prompt = PromptTemplate(input_variables=["catalog"], template=prompt_text + "\nCatalogo: {catalog}")
            # Use pipe syntax to avoid LLMChain deprecation
            chains[key] = prompt | llm
    return chains

# Metric calculations on toy catalog
def precision_at_k(recommended_ids, relevant_ids, k=3):
    hits = len([i for i in recommended_ids[:k] if i in relevant_ids])
    return hits / k if k else 0

def coverage(recommended_ids, all_ids):
    return len(set(recommended_ids)) / len(all_ids) if all_ids else 0

# Asynchronous aggregation of model outputs
async def generate_recommendations(chains, catalog_str):
    async def run_chain(name, ch):
        result = await ch.ainvoke({"catalog": catalog_str})
        # handle both dict and message object
        text = result["content"] if isinstance(result, dict) else result.content
        # parse digits as IDs
        ids = [int(x) for x in text.split() if x.isdigit()]
        return name, ids

    tasks = [run_chain(name, chain) for name, chain in chains.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)

# Evaluator LLM chain
EVALUATOR_TEMPLATE = PromptTemplate(
    input_variables=["outputs", "metrics"],
    template=("Ricevi queste liste: {outputs}\n"  
              "E questi valori di metriche quantitative: {metrics}\n"  
              "Fornisci la lista finale ottimizzata bilanciando le metriche." )
)

def build_evaluator_chain():
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", **COMMON_LLM_PARAMS)
    return EVALUATOR_TEMPLATE | evaluator_llm

# Main experimental loop
async def main():
    chains = build_model_chains()
    evaluator = build_evaluator_chain()
    report = []
    catalog_str = ", ".join([f"{item['id']}:{item['name']}" for item in TOY_CATALOG])

    for metric in PROMPT_VARIANTS:
        # filter chains by metric suffix
        subset = {k:v for k,v in chains.items() if k.endswith(metric)}
        outputs = await generate_recommendations(subset, catalog_str)

        # quantitative metrics
        all_ids = [item['id'] for item in TOY_CATALOG]
        rec_ids = [rid for lst in outputs.values() for rid in lst]
        metrics = {
            "precision@3": precision_at_k(rec_ids, all_ids, k=3),
            "coverage": coverage(rec_ids, all_ids)
        }

        # evaluator
        eval_input = {"outputs": json.dumps(outputs), "metrics": json.dumps(metrics)}
        eval_result = await evaluator.ainvoke(eval_input)
        final_list = eval_result["content"] if isinstance(eval_result, dict) else eval_result.content

        report.append({
            "metric": metric,
            "raw_outputs": outputs,
            "quantitative": metrics,
            "final": final_list
        })

    fname = f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {fname}")

if __name__ == "__main__":
    asyncio.run(main())
