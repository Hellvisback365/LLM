import asyncio
import os
from langchain_openai import ChatOpenAI  # Import aggiornato
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

# Carica le variabili dal file .env
load_dotenv()

# Recupera la chiave API dall'ambiente
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("La variabile d'ambiente OPENROUTER_API_KEY non è impostata.")

# --- Configurazione Centralizzata ---
COMMON_LLM_PARAMS = {
    "openai_api_base": "https://openrouter.ai/api/v1",
    "openai_api_key": openrouter_api_key,
    "temperature": 0.7,
    "max_tokens": 2048,
}

MODEL_CONFIGS = [
    {"name": "llama_4_maverick", "model_id": "meta-llama/llama-4-maverick:free"},
    {"name": "deepseek_chat_v3", "model_id": "deepseek/deepseek-chat-v3-0324:free"},
    {"name": "mistral_small_3.1", "model_id": "mistralai/mistral-small-3.1-24b-instruct:free"},
]

# Definizione del prompt condiviso
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="Rispondi al seguente prompt in modo dettagliato: {user_input}"
)

# --- Creazione dinamica delle catene usando la sintassi | (pipe) ---
chains = {}
for config in MODEL_CONFIGS:
    try:
        chat_model = ChatOpenAI(
            model=config["model_id"],
            **COMMON_LLM_PARAMS
        )
        # Crea la catena con la nuova sintassi
        chains[config["name"]] = prompt_template | chat_model
    except Exception as e:
        print(f"Errore durante l'inizializzazione del modello {config['name']}: {e}")

# --- Funzione Asincrona per aggregare le risposte ---
async def aggregate_llm_outputs_async(prompt: str) -> dict:
    tasks = []
    for model_name, chain in chains.items():
        async def run_chain_safely(name, ch, p):
            try:
                # Usa ainvoke invece di arun
                response = (await ch.ainvoke({"user_input": p})).content
                return name, response
            except Exception as e:
                print(f"Errore durante l'esecuzione della catena {name}: {e}")
                return name, f"Errore: {e}"

        tasks.append(run_chain_safely(model_name, chain, prompt))

    results = await asyncio.gather(*tasks)
    return {name: response for name, response in results}
 
# Esecuzione di esempio asincrona
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
 
Titolo e Anno.
Breve sinossi (1-2 frasi).
Come si collega alle mie preferenze (specialmente quelle importanti).
Perché pensi sia adatto al mio stato d'animo attuale."""
 
    print("Avvio chiamate LLM in parallelo...")
    outputs = await aggregate_llm_outputs_async(user_prompt)
    print("Chiamate LLM completate.")
 
    print("\n--- Risposte dei Modelli ---")
    for model, response in outputs.items():
        print(f"Output da {model}:\n{response}\n{'-' * 50}")
 
if __name__ == "__main__":
    # Esegue la funzione main asincrona
    asyncio.run(main())