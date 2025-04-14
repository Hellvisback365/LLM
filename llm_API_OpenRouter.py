import asyncio
import os
import json # salvataggio dati
from datetime import datetime
from dotenv import load_dotenv

# Import per LangChain
from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Carica le variabili dal file .env
load_dotenv()

# Recupera la chiave API dall'ambiente
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("La variabile d'ambiente OPENROUTER_API_KEY non è impostata.")

# Configurazione Centralizzata
COMMON_LLM_PARAMS = {
    "openai_api_base": "https://openrouter.ai/api/v1",
    "openai_api_key": openrouter_api_key,
    "temperature": 0.7,
    "max_tokens": 2048,
}

MODEL_CONFIGS = [
    {"name": "llama_4_maverick", "model_id": "meta-llama/llama-4-maverick:free"},
    {"name": "DeepSeek_R1", "model_id": "deepseek/deepseek-r1:free"},
    {"name": "Gemma_3_27B", "model_id": "google/gemma-3-27b-it:free"},
]

# Definizione del prompt condiviso
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="Rispondi al seguente prompt in modo dettagliato e identato: {user_input}"
)

# Creazione dinamica delle catene usando la sintassi pipe
chains = {}
for config in MODEL_CONFIGS:
    try:
        chat_model = ChatOpenAI(
            model=config["model_id"],
            **COMMON_LLM_PARAMS
        )
        # Costruisco la catena, passando il prompt al modello
        chains[config["name"]] = prompt_template | chat_model
    except Exception as e:
        print(f"Errore durante l'inizializzazione del modello {config['name']}: {e}")

# Funzione Asincrona per aggregare le risposte dai modelli 
async def aggregate_llm_outputs_async(prompt: str) -> dict:
    tasks = []
    for model_name, chain in chains.items():
        async def run_chain_safely(name, ch, p):
            try:
                # metodo asincrono ainvoke
                response = (await ch.ainvoke({"user_input": p})).content
                return name, response
            except Exception as e:
                print(f"Errore durante l'esecuzione della catena {name}: {e}")
                return name, f"Errore: {e}"
        tasks.append(run_chain_safely(model_name, chain, prompt))

    results = await asyncio.gather(*tasks)
    return {name: response for name, response in results}
 
 # Funzione per salvare l'input e l'output in un file JSON 
def save_result(log_entry, filename="llm_results.json"):
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
    except Exception:
        logs = []
    logs.append(log_entry)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    print(f"Risultato salvato su {filename}.")
 
# Esecuzione di esempio asincrona
async def main():
    user_prompt = """ Considerando le preferenze: Azione (importante), Adrenalinico (importante), Simili a Mad Max: Fury Road o The Covenant, Moderni (2000+), No <1970.

    Consiglia semplicemente i titoli di 3 film adatti. **Non includere Mad Max: Fury Road o The Covenant nella lista.** Nessun'altra informazione richiesta. Sii estremamente conciso.
    """ # ( PROMPT 3 modificato per evitare ripetizioni di film già visti )
    
    
    #----------------------------------------------------------------
    # """ Considerando le preferenze: Azione (importante), Adrenalinico (importante), Simili a Mad Max/Covenant, Moderni (2000+), No <1970.

    # **Consiglia semplicemente i titoli di 3 film adatti.** Nessun'altra informazione richiesta. Sii estremamente conciso.
    # """     # ( PROMPT 3 ) 
    
    
    #----------------------------------------------------------------
    #  """ Basandomi su: Azione (molto importante), Mood Adrenalinico (molto importante), Simili a Mad Max/Covenant, Ambientazione Moderna (2000+).

    # **Qual è il SINGOLO film migliore che consiglieresti? Fornisci solo Titolo e Anno.** Risposta brevissima.
    # """   # ( PROMPT 4 )
    
    
    #----------------------------------------------------------------
    # """ Sto cercando consigli per dei film da guardare. Basati sulle seguenti informazioni:
    # - Generi: Azione. (Più importante)
    # - Film Simili Apprezzati: Mad Max: Fury Road, The Covenant.
    # - Ambientazioni Desiderate: Anni 2000-2025, Moderno. (Secondario)
    # - Attori Graditi: Keanu Reeves, Johnny Depp. (Bonus)
    # - Stato d'Animo: Cerco qualcosa che sia principalmente adrenalinico. (Più importante)
    # - Vincoli: Evita film più vecchi del 1970.

    # Forniscimi una lista di 3 film. **Per ciascun film, indica SOLO Titolo e Anno.** Risposta molto breve e diretta.
    # """     # ( PROMPT 2 ) 
        
        
    #----------------------------------------------------------------    
    # """ Sto cercando consigli per dei film da guardare.
    # Per favore, basati sulle seguenti informazioni, dando priorità agli elementi contrassegnati come più importanti.
    
    # Generi: Azione. (Più importante)
    # Film Simili Apprezzati: Mad Max: Fury Road, The Covenant.
    # Ambientazioni Desiderate: Anni 2000-2025, Moderno. (Bello se presente, ma secondario rispetto al genere)
    # Attori Graditi: Keanu Reeves, Johnny Depp. (Un bonus, non una necessità assoluta)
    # Stato d'Animo: Riflessivo. Cerco qualcosa che sia principalmente adrenalinico. (Più importante)
    # Vincoli: Evita film più vecchi del 1970.
    # Forniscimi una lista di 3 film. Per ciascuno, indica:
    
    # Titolo e Anno.
    # Breve sinossi (1-2 frasi).
    # Come si collega alle mie preferenze (specialmente quelle importanti).
    # Perché pensi sia adatto al mio stato d'animo attuale. """   # ( PROMPT 1 ) 
                    
                    
                    
    print("Avvio chiamate LLM in parallelo...")
    outputs = await aggregate_llm_outputs_async(user_prompt)
    print("Chiamate LLM completate.")
 
    print("\n--- Risposte dei Modelli ---")
    for model, response in outputs.items():
        print(f"Output da {model}:\n{response}\n{'-' * 50}")
 
 # Log da salvare: include l'input, gli output e il timestamp
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_prompt": user_prompt,
        "outputs": outputs
    }
    save_result(log_entry) 
 
if __name__ == "__main__":
    # Esegue la funzione main asincrona
    asyncio.run(main())