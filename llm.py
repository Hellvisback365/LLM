import asyncio
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline


load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

COMMON_LLM_PARAMS = {
    "base_url": "https://openrouter.ai/api/v1",
    "temperature": 0.7,
    "max_tokens": 2048,
}

# --- Prompt Template ---
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="""Rispondi al seguente prompt in modo dettagliato:

{user_input}
"""
)

# --- Modelli OpenRouter ---
MODEL_CONFIGS = [
    {"name": "llama_4_maverick", "model_id": "meta-llama/llama-4-maverick:free"},
    {"name": "deepseek_chat_v3", "model_id": "deepseek/deepseek-chat-v3-0324:free"},
    {"name": "mistral_small_3.1", "model_id": "mistralai/mistral-small-3.1-24b-instruct:free"},
]

# --- Modelli locali HuggingFace ---
LOCAL_MODELS = [
    {
        "name": "mistral_local",
        "model_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "device": "cuda"
    },
    {
        "name": "gemma_local",
        "model_path": "google/gemma-2b-it",
        "device": "cuda"
    }
]

chains = {}

# --- Inizializzazione modelli OpenRouter ---
for config in MODEL_CONFIGS:
    try:
        chat_model = ChatOpenAI(
            model=config["model_id"],
            openai_api_key=openrouter_api_key,
            **COMMON_LLM_PARAMS
        )
        chains[config["name"]] = {"user_input": RunnablePassthrough()} | prompt_template | chat_model
    except Exception as e:
        print(f"Errore inizializzazione modello {config['name']}: {e}")

# --- Inizializzazione modelli locali ---
for local in LOCAL_MODELS:
    try:
        tokenizer = AutoTokenizer.from_pretrained(local["model_path"])
        model = AutoModelForCausalLM.from_pretrained(local["model_path"])

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )

        hf_llm = HuggingFacePipeline(pipeline=pipe)
        chains[local["name"]] = {"user_input": RunnablePassthrough()} | prompt_template | hf_llm

    except Exception as e:
        print(f"Errore inizializzazione modello locale {local['name']}: {e}")


# --- Funzione per eseguire tutte le catene in parallelo ---
async def aggregate_llm_outputs_async(prompt: str) -> dict:
    async def run_chain(name: str, ch: Runnable, prompt: str):
        try:
            result = await ch.ainvoke({"user_input": prompt})
            return name, result.content if hasattr(result, "content") else result
        except Exception as e:
            return name, f"Errore: {e}"

    tasks = [run_chain(name, chain, prompt) for name, chain in chains.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)

# --- Funzione principale ---
async def main():
    user_prompt = """Sto cercando consigli per dei film da guardare.
Per favore, basati sulle seguenti informazioni, dando priorità agli elementi contrassegnati come più importanti.

Generi: Azione. (Più importante)
Film Simili Apprezzati: Mad Max: Fury Road, The Covenant.
Ambientazioni Desiderate: Anni 2000-2025, Moderno. (Bello se presente, ma secondario rispetto al genere)
Attori Graditi: Keanu Reeves, Johnny Depp. (Un bonus, non una necessità assoluta)
Stato d'Animo: Riflessivo. Cerco qualcosa che sia principalmente adrenalinico. (Più importante)
Vincoli: Evita film più vecchi del 1970.
Forniscimi una lista di 3 film. Per ciascuno, indica:
- Titolo e Anno.
- Breve sinossi (1-2 frasi).
- Come si collega alle mie preferenze (specialmente quelle importanti).
- Perché pensi sia adatto al mio stato d'animo attuale.
"""

    print("Avvio chiamate LLM in parallelo...")
    outputs = await aggregate_llm_outputs_async(user_prompt)
    print("Chiamate LLM completate.\n")

    print("--- Risposte dei Modelli ---")
    for model, response in outputs.items():
        print(f"\nOutput da {model}:\n{response}\n{'-' * 60}")

if __name__ == "__main__":
    asyncio.run(main())
