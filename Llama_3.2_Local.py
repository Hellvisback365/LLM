import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# Importazioni aggiornate:
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Configurazione del modello locale Llama-3.2-1B-Instruct ---
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    early_stopping=True,     # Interrompe la generazione quando una sequenza completa viene riconosciuta
    truncation=True  # Aggiunto per gestire la lunghezza e i warning sulla troncatura
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = "Scrivi un paragrafo sul tema: {topic}"
prompt = PromptTemplate(template=prompt_template, input_variables=["topic"])

chain = LLMChain(llm=llm, prompt=prompt)

input_topic = "L'evoluzione dell'intelligenza artificiale applicata alla medicina"

outputs = []
num_runs = 3

print("Generazione output dalle tre chiamate al modello...\n")
for i in range(num_runs):
    print(f"Esecuzione {i+1}/{num_runs}...")
    # Utilizza chain.invoke invece di chain.run (metodo deprecato)
    output = chain.invoke({"topic": input_topic})
    outputs.append(output)
    print(f"Output {i+1}:\n{output}\n{'-'*40}\n")

# Aggregazione dei risultati estraendo il campo 'text' da ciascun output
aggregated_output = "\n---\n".join([output['text'] for output in outputs])
print("Output Aggregato:\n")
print(aggregated_output)
