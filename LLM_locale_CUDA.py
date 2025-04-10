import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel
import time

# Verifica CUDA
print(f"CUDA disponibile: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Memoria VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Definizione di tre modelli diversi (tutti piccoli per la tua GPU da 4GB)
model_configs = [
    {"name": "google/gemma-2b-it", "desc": "Modello Google Gemma 2B"},
    {"name": "facebook/opt-125m", "desc": "OPT piccolo di Meta"},
    {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "desc": "TinyLlama chat"}
]

# Funzione per caricare un modello
def load_model(model_id):
    print(f"Caricamento del modello: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Gestione dei diversi tipi di tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,  # Attiva campionamento per temperatura e top_p
        top_p=0.95
    )
    
    return HuggingFacePipeline(pipeline=pipe)

# Caricare i modelli uno alla volta per risparmiare memoria
models = {}
for config in model_configs:
    models[config["name"]] = load_model(config["name"])
    # Libera memoria CUDA dopo aver caricato ciascun modello
    torch.cuda.empty_cache()
    print(f"Memoria GPU usata: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# Definizione dei prompt template
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="Domanda: {query}\n\nRisposta:"
)

# Funzione per pulire l'output
def clean_output(text):
    # Rimuovi tag e formattazione non necessaria
    # Adatta in base all'output specifico dei tuoi modelli
    import re
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'Domanda:.*?\n\nRisposta:', '', text)
    text = re.sub(r'<userStyle>.*?</userStyle>', '', text)
    return text.strip()

# Funzione per generare output dai tre modelli
def generate_from_models(query):
    results = {}
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    for name, model in models.items():
        print(f"Generazione da {name}...")
        start_time = time.time()
        
        # Carica il modello sulla GPU prima dell'inferenza
        if hasattr(model.pipeline.model, "to"):
            model.pipeline.model.to("cuda")
        
        try:
            chain = prompt_template | model
            response = chain.invoke({"query": query})
            
            # Pulisci l'output
            cleaned_response = clean_output(response)
            
            end_time = time.time()
            results[name] = {
                "raw": response,
                "clean": cleaned_response,
                "time": end_time - start_time
            }
            print(f"Completato in {end_time - start_time:.2f} secondi")
        except Exception as e:
            print(f"Errore con il modello {name}: {e}")
            results[name] = {"error": str(e)}
        
        # Libera memoria GPU dopo l'uso
        if hasattr(model.pipeline.model, "to"):
            model.pipeline.model.to("cpu")
        torch.cuda.empty_cache()
    
    return results

# Test con una query
results = generate_from_models("Spiega brevemente cosa sono i sistemi di raccomandazione e come funzionano")

# Visualizza i risultati
print("\n=== RISULTATI ===")
for name, result in results.items():
    print(f"\nModello: {name}")
    if "clean" in result:
        print(f"Tempo: {result['time']:.2f} secondi")
        print(f"Risposta:\n{result['clean']}")
    else:
        print(f"Errore: {result['error']}")
    print("-" * 50)