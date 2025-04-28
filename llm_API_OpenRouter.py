# ----------------------------
# Import e setup iniziale
# ----------------------------
import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# ----------------------------
# LangChain imports
# ----------------------------
from langchain.prompts import PromptTemplate # Importa il costruttore di prompt strutturati di LangChain.
from langchain_openai import ChatOpenAI # Consente di istanziare un LLM chat-based con parametri personalizzati.
from langchain.output_parsers import ResponseSchema, StructuredOutputParser # Garantisce che l’output dell’LLM rispetti un formato JSON atteso.

# ----------------------------
# 1. Setup ambiente e parametri
# ----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in environment")

COMMON_LLM_PARAMS = {
    "openai_api_base": "https://openrouter.ai/api/v1",
    "openai_api_key": OPENROUTER_API_KEY,
    "temperature": 0.7, # grado di casualità nelle risposte (0.7 = moderata creatività).
    "max_tokens": 512,
}
LLM_MODEL_ID = "mistralai/mistral-7b-instruct:free"

# ----------------------------
# 2. Toy catalog
# ----------------------------
TOY_CATALOG = [ # lista di dizionari, ciascuno rappresenta un prodotto food con id e name.
    {"id": 1, "name": "Margherita Pizza"},
    {"id": 2, "name": "Sushi Roll"},
    {"id": 3, "name": "Falafel Wrap"},
    {"id": 4, "name": "Veggie Burger"},
    {"id": 5, "name": "Pad Thai"},
    {"id": 6, "name": "Tacos al Pastor"},
    {"id": 7, "name": "Caesar Salad"},
    {"id": 8, "name": "Ramen Bowl"},
]
CATALOG_STR = ", ".join(f"{item['id']}:{item['name']}" for item in TOY_CATALOG) # stringa formattata per il prompt
ALL_IDS = [item["id"] for item in TOY_CATALOG] # lista di tutti gli ID; usata per calcolare metriche come coverage o precision@k.
 
# ----------------------------
# 3. Metric-driven prompts
# ----------------------------
PROMPT_VARIANTS = { # definire tre varianti di prompt, ciascuna ottimizzata su una metrica diversa. 
                    # Ogni voce è una tupla (testo_istruzione, esempio_output_atteso).
                    # Collegamento: queste varianti alimentano make_prompt e poi build_prompt_chains.
                   
    "accuracy": ( # generare raccomandazioni focalizzate su accuratezza percepita.
        "Suggerisci i top 3 prodotti food più rilevanti, massimizzando l'accuratezza.",
        "[1, 4, 3]"
    ),
    "precision_at_3": ( # precisione sui primi 3 suggerimenti.
        "Suggerisci 3 ID separati da virgola che l'utente apprezzerebbe con certezza.",
        "[2, 3, 4]"
    ),
    "coverage": ( # massimizzare la copertura diversificando categorie.
        "Scegli esattamente 3 ID da 3 categorie diverse e rispondi separati da virgola.",
        "[2, 3, 6]"
    ),
}

# Costruzione del singolo prompt e parser
def make_prompt(metric_key: str, metric_text: str, example: str = None):
    schema = ResponseSchema( # Definisco lo schema di output desiderato per il parser
        name="ids",
        description="Lista JSON di soli ID numerici delle raccomandazioni, in ordine di priorità."
    )
    parser = StructuredOutputParser.from_response_schemas([schema]) # Crea un parser strutturato per l'output dell'LLM.
    # prendo le vere istruzioni di formato dal parser e escapo le graffe
    fmt_instr = parser.get_format_instructions().replace("{", "{{").replace("}", "}}") # per evitare conflitti con le graffe del template.
    
    # antepongo un esempio per chiarire le istruzioni 
    if example:
        fmt_instr = f"Esempio corretto: {example}\n" + fmt_instr

    # costruisco il template del prompt per l'LLM.
    template_str = (
        f"Metric: {metric_key}\n"
        f"Descrizione: {metric_text}\n"
        f"Catalogo: {{catalog}}\n"
        f"{fmt_instr}"
    )
    return PromptTemplate(input_variables=["catalog"], template=template_str), parser # ritorno sia il template del prompt che il parser strutturato.
                                                                                      # usati da build_prompt_chains.
    
# ----------------------------
# 4. Costruzione delle catene
# ----------------------------
def build_prompt_chains(): # costruisce le catene per ogni metrica.
    llm = ChatOpenAI(model=LLM_MODEL_ID, **COMMON_LLM_PARAMS) # modello LLM da usare per la generazione.
    chains, parsers, raw_prompts = {}, {}, {} # dizionari per memorizzare le catene, i parser e i prompt raw.
    for metric, (text, example) in PROMPT_VARIANTS.items(): # ciclo su ogni metrica definita in PROMPT_VARIANTS.
        prompt, parser = make_prompt(metric, text, example) # costruisco il prompt e il parser per la metrica corrente.
        chains[metric] = prompt | llm # creo la catena combinando il prompt e il modello LLM.
        parsers[metric] = parser # memorizzo il parser per questa metrica.
        raw_prompts[metric] = prompt.template # memorizzo il prompt raw per questa metrica.
    return chains, parsers, raw_prompts # ritorno le catene, i parser e i prompt raw.

def build_evaluator_chain(): # costruisce la catena per il valutatore multi-run.
    # STESSO modello LLM della generazione per la valutazione
    llm = ChatOpenAI(model=LLM_MODEL_ID, **COMMON_LLM_PARAMS) # modello LLM da usare per la valutazione.
 
    # Definisco lo schema di output desiderato per il valutatore multi-run.
    # Includo due schemi: uno per gli ID finali e uno per la giustificazione.
    final_ids_schema = ResponseSchema( 
        name="final_ids", 
        description="Lista JSON ottimizzata finale di massimo 3 ID numerici, basata sull'analisi di tutti i run." 
    )
    justification_schema = ResponseSchema(
        name="justification",
        description="Breve spiegazione testuale del perché questi ID sono stati scelti come raccomandazione finale, considerando i risultati (output e metriche) dei diversi run."
    )
    parser = StructuredOutputParser.from_response_schemas([final_ids_schema, justification_schema]) # Crea un parser strutturato per l'output del valutatore.
    fmt_instr = parser.get_format_instructions().replace("{","{{").replace("}","}}")

    # Crea il template del prompt per il valutatore multi-run
    prompt = PromptTemplate(
        # Input: una stringa JSON contenente i risultati di tutti i run
        input_variables=["multi_run_results_json"], 
        template=(
            "Sei un analista esperto di sistemi di raccomandazione.\n"
            "Ti vengono forniti i risultati di {num_runs} esecuzioni indipendenti (run) dello stesso processo di generazione di raccomandazioni.\n"
            "Per ogni run, sono stati generati suggerimenti per diverse metriche (`accuracy`, `precision_at_3`, `coverage`) e sono state calcolate le performance (`precision@3`, `coverage`).\n\n"
            "Dati dei Run:\n"
            "```json\n"
            "{multi_run_results_json}\n"
            "```\n\n"
            "ISTRUZIONI:\n"
            "1. Analizza i risultati (`outputs`) e le metriche (`metrics`) di tutti i run.\n"
            "2. Considera la consistenza dei suggerimenti tra i run e le performance medie o migliori osservate.\n"
            "3. Basandoti su questa analisi complessiva, proponi una singola lista finale ottimizzata contenente al massimo 3 ID numerici unici.\n"
            "4. Fornisci una breve giustificazione testuale per la tua scelta finale, spiegando come sei arrivato a quella lista basandoti sui dati forniti.\n\n"
            "Formato di output richiesto:\n"
            f"{fmt_instr}" # Includo le istruzioni di formattazione JSON
        )
    )
    # Includo num_runs nel partial_variables per usarlo nel template
    prompt = prompt.partial(num_runs=3) # Assumendo 3 run come nell'esempio

    return prompt | llm, parser # ritorno sia la catena del valutatore che il parser strutturato.


# ----------------------------
# 5. Funzioni di utilità
# ----------------------------
def precision_at_k(recs, rel, k=3): # calcola la precisione a k. conta quanti dei primi k consigli sono rilevanti (rel).
    return sum(1 for i in recs[:k] if i in rel) / k if k else 0 # Se k è 0, ritorna 0. 

def coverage_metric(recs, all_ids): # calcola la copertura. percentuale di elementi unici raccomandati rispetto al totale.
    return len(set(recs)) / len(all_ids) if all_ids else 0 # Se all_ids è vuoto, ritorna 0.

# Scopo: funzione asincrona che, dato una chain LLM e un parser, invia il prompt, riceve la risposta, la pulisce e restituisce: 
# cleaned: lista di interi (ID raccomandati).
# raw: anteprima del contenuto grezzo (debug).
async def generate_for_metric(chain, parser, catalog_str): 
    # Invoca la catena LLM per generare raccomandazioni basate sul catalogo fornito.
    try: 
        result = await chain.ainvoke({"catalog": catalog_str}) # Passa il catalogo come input alla catena.
    except Exception as e:
        print(f"LLM error for {parser.name}: {e}")
        return [], ""
    content = getattr(result, "content", result) # Ottengo il contenuto della risposta LLM.
    raw = "\n".join(content.splitlines()[:5]) + ("..." if len(content.splitlines()) > 5 else "") # Anteprima del contenuto grezzo (primi 5 linee).
    # Provo a parsare il contenuto come JSON strutturato.
    try:
        ids = parser.parse(content).get("ids", []) # Estraggo gli ID raccomandati dal parser.
    except Exception: # Se il parser fallisce, provo a caricare il contenuto come JSON.
        try:
            ids = json.loads(content) # Provo a caricare il contenuto come JSON.
        except Exception: 
            ids = [] # Se fallisce, assegno una lista vuota.
    # Se ids è ancora stringa, lo carico come JSON
    if isinstance(ids, str): 
        try:
            ids = json.loads(ids) # Provo a caricare la stringa come JSON.
        except:
            ids = [] # Se fallisce, assegno una lista vuota.
    # Cast a int e pulisco
    cleaned = [] # Lista per memorizzare gli ID puliti.
    for x in ids: # Itero su ogni ID raccomandato.
        try:
            cleaned.append(int(x)) # Provo a convertire l'ID in un intero.
        except:
            pass # Se fallisce, ignoro l'ID.
    return cleaned, raw # Ritorno la lista di ID puliti e l'anteprima del contenuto grezzo.

# ----------------------------
# 6. Main: generazione, metriche e salvataggio report
# ----------------------------
def main(): # Inizio della funzione principale.
    chains, parsers, prompts = build_prompt_chains() # Costruisco le catene per ogni metrica.
    
    evaluator_chain, evaluator_parser = build_evaluator_chain() # Costruisco il valutatore (ora uso Mistral e il nuovo prompt)

    all_ids = ALL_IDS 
    num_runs = 3 
    all_run_results = [] # Lista per memorizzare i risultati di tutti i run.

    print(f"Starting {num_runs} generation runs using {LLM_MODEL_ID}...") # Inizio del ciclo principale per i run di generazione.

    # CICLO ESTERNO PER I RUNS 
    for run_number in range(1, num_runs + 1): # Itero su ogni run da 1 a num_runs.
        print(f"\n--- STARTING RUN {run_number}/{num_runs} ---") # Inizio del ciclo per il run corrente.
        run_outputs = {} # Dizionario per memorizzare gli output di questo run.
        run_raw_outputs = {} # Dizionario per memorizzare gli output raw di questo run.
        run_metrics = {} # Dizionario per memorizzare le metriche di questo run.

        # CICLO INTERNO PER LE METRICHE 
        print(f"  [Run {run_number}] Generating recommendations...") # Inizio del ciclo per le metriche.
        for metric in PROMPT_VARIANTS: # Itero su ogni metrica definita in PROMPT_VARIANTS.
            # (stampa e chiamata a generate_for_metric come prima)
            ids, raw = asyncio.run(generate_for_metric(chains[metric], parsers[metric], CATALOG_STR)) # Invoco la funzione per generare raccomandazioni per la metrica corrente.
            print(f"    Metric: {metric.upper()} -> Parsed IDs: {ids}") # Stampo gli ID raccomandati per la metrica corrente.
            # print(f"    Raw: {raw}") # Opzionale: decommenta per vedere il raw
            run_outputs[metric] = ids # Memorizzo gli ID raccomandati per questa metrica.
            run_raw_outputs[metric] = raw # Memorizzo il contenuto raw per questa metrica.

        # CALCOLO METRICHE PER QUESTO RUN
        print(f"  [Run {run_number}] Calculating metrics...") # Inizio del ciclo per calcolare le metriche.
        for m, recs in run_outputs.items(): # Itero su ogni metrica e le raccomandazioni generate.
            run_metrics[m] = { # Dizionario per memorizzare le metriche calcolate.
                "precision@3": precision_at_k(recs, all_ids), # Calcolo la precisione a 3.
                "coverage": coverage_metric(recs, all_ids)  # Calcolo la copertura.
            }
        print(f"    Metrics for Run {run_number}: {run_metrics}") # Stampo le metriche calcolate per questo run.

        # Salva i risultati completi di questo run
        all_run_results.append({  # Aggiungo i risultati di questo run alla lista totale.
            "run_number": run_number, # Numero del run corrente.
            "outputs": run_outputs, # Raccomandazioni generate per ogni metrica.
            "raw_outputs": run_raw_outputs, # Potrebbe essere utile per il debug
            "metrics": run_metrics # Metriche calcolate per questo run.
        })
        print(f"--- COMPLETED RUN {run_number}/{num_runs} ---") # Fine del ciclo interno per le metriche.


    # VALUTAZIONE COMPLESSIVA POST-RUN
    print(f"\n--- STARTING OVERALL EVALUATION (using {LLM_MODEL_ID}) ---") # Inizio della valutazione complessiva.
    final_recommendation_ids = [] # Lista per memorizzare gli ID raccomandati finali.
    final_justification = "Evaluation could not be performed or parsed." # Giustificazione predefinita in caso di errore.

    if all_run_results: # Procedi solo se ci sono risultati dai run
        try:
            # Serializza tutti i risultati dei run in un JSON per il prompt
            all_run_results_json = json.dumps(all_run_results, ensure_ascii=False) 

            # Invoca la catena del valutatore
            eval_res = asyncio.run(     
                evaluator_chain.ainvoke({  
                    "multi_run_results_json": all_run_results_json   # Passa i risultati dei run come input alla catena.
                })
            )
            print("  Evaluator Raw Response:", eval_res.content) # Utile per debug

            # Tenta di parsare l'output strutturato (final_ids, justification)
            try:
                parsed_eval = evaluator_parser.parse(eval_res.content) 
                raw_final_ids = parsed_eval.get("final_ids", [])  # Estraggo gli ID finali raccomandati dal parser.
                final_justification = parsed_eval.get("justification", "No justification provided by evaluator.") # Estraggo la giustificazione finale dal parser.

                # Pulisci e de-duplica gli ID finali
                # Gestisce il caso in cui l'LLM restituisca una stringa JSON invece di una lista
                if isinstance(raw_final_ids, str):
                   try:
                       raw_final_ids = json.loads(raw_final_ids) # Provo a caricare la stringa come JSON.
                   except json.JSONDecodeError:
                       print("  Warning: final_ids from evaluator was a string but not valid JSON.")   # Se non è JSON valido, assegno una lista vuota.
                       raw_final_ids = []

                if isinstance(raw_final_ids, list): # Controllo se raw_final_ids è una lista
                    seen = set() # Set per tenere traccia degli ID già visti
                    uniq_final = [] # Lista per memorizzare gli ID unici finali
                    for x in raw_final_ids: # Itero su ogni ID finale raccomandato.
                        try:
                            i = int(x) # Provo a convertire l'ID in un intero.
                            if i not in seen: # Se l'ID non è già stato visto, lo aggiungo alla lista finale.
                                seen.add(i) # Aggiungo l'ID al set dei visti
                                uniq_final.append(i) # Aggiungo l'ID alla lista finale
                        except (ValueError, TypeError): # Se non riesco a convertire l'ID in un intero, lo ignoro.
                             print(f"  Warning: Could not convert item '{x}' to int in final_ids.") # Ignoro l'ID.
                    final_recommendation_ids = uniq_final[:3] # Limita a 3 ID
                else: # Se raw_final_ids non è una lista, assegno una lista vuota.
                    print("  Warning: Parsed final_ids was not a list.") # Assegno una lista vuota.
                    final_recommendation_ids = [] # Nessun ID recuperato


            except Exception as parse_error: # Se il parser fallisce, gestisco l'eccezione.
                print(f"  Error parsing evaluator response: {parse_error}. Falling back.") # Assegno una lista vuota.
                final_justification = f"Failed to parse structured output. Raw: {eval_res.content}" # Nessun ID recuperato.
                # Fallback: tenta di estrarre almeno gli ID se il formato JSON è semplice
                try:
                    simple_json = json.loads(eval_res.content) 
                    raw_final_ids = simple_json.get("final_ids", []) 
                    if isinstance(raw_final_ids, str): raw_final_ids = json.loads(raw_final_ids) # controllo delle stringhe nidificate
                    if isinstance(raw_final_ids, list): # Controllo se raw_final_ids è una lista
                      seen = set() # Set per tenere traccia degli ID già visti
                      uniq_final = [] # Lista per memorizzare gli ID unici finali
                      for x in raw_final_ids: # Itero su ogni ID finale raccomandato.
                           try:
                              i = int(x)
                              if i not in seen:
                                  seen.add(i)
                                  uniq_final.append(i)
                           except (ValueError, TypeError): pass
                      final_recommendation_ids = uniq_final[:3]
                    else: final_recommendation_ids = []
                except Exception: final_recommendation_ids = [] # Nessun ID recuperato

        except Exception as eval_error: # Se la chiamata al valutatore fallisce, gestisco l'eccezione.
            print(f"  Error during evaluator LLM call: {eval_error}") # Assegno una lista vuota.
            # final_recommendation_ids resta []
            # final_justification resta il messaggio di errore predefinito

    print(f"  Final Recommended IDs: {final_recommendation_ids}") # Stampo gli ID raccomandati finali.
    print(f"  Justification: {final_justification}") # Stampo la giustificazione finale.
    print("--- EVALUATION COMPLETE ---") # Fine della valutazione complessiva.

    # --- SALVATAGGIO REPORT FINALE ---
    report = { # Creo un report finale in formato JSON.
        "generation_model": LLM_MODEL_ID, # Specifica il modello usato per generare le raccomandazioni
        "evaluation_model": LLM_MODEL_ID, # Specifica il modello usato per valutare
        "num_runs": num_runs, # Numero di run eseguiti
        "catalog_used": TOY_CATALOG, # Aggiungiamo il catalogo per riferimento
        "metric_prompts": prompts, # I prompt usati per la generazione
        "run_results": all_run_results, # Risultati dettagliati di ogni run
        "overall_recommendation": { # Sezione per la raccomandazione finale
            "ids": final_recommendation_ids, # ID raccomandati finali
            "justification": final_justification # Giustificazione finale
        }
    }
    filename = f"report_multi_run_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json" # Nome del file per il report finale.
    with open(filename, 'w', encoding='utf-8') as f: # Apro il file in modalità scrittura.
        json.dump(report, f, indent=2, ensure_ascii=False) # Serializzo il report in formato JSON e lo scrivo nel file.
    print(f"\nMulti-run report with evaluation saved to {filename}") # Stampo il percorso del file salvato.

if __name__ == "__main__": # Se il file viene eseguito come script principale, chiama la funzione main.
    main() # Esegui la funzione principale.