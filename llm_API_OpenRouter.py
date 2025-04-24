import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

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
    "temperature": 0.7,
    "max_tokens": 512,
}
LLM_MODEL_ID = "mistralai/mistral-7b-instruct:free"

# ----------------------------
# 2. Toy catalog
# ----------------------------
TOY_CATALOG = [
    {"id": 1, "name": "Margherita Pizza"},
    {"id": 2, "name": "Sushi Roll"},
    {"id": 3, "name": "Falafel Wrap"},
    {"id": 4, "name": "Veggie Burger"},
    {"id": 5, "name": "Pad Thai"},
    {"id": 6, "name": "Tacos al Pastor"},
    {"id": 7, "name": "Caesar Salad"},
    {"id": 8, "name": "Ramen Bowl"},
]
CATALOG_STR = ", ".join(f"{item['id']}:{item['name']}" for item in TOY_CATALOG)
ALL_IDS = [item["id"] for item in TOY_CATALOG]

# ----------------------------
# 3. Metric-driven prompts
# ----------------------------
PROMPT_VARIANTS = {
    "accuracy": (
        "Suggerisci i top 3 prodotti food più rilevanti, massimizzando l'accuratezza.",
        "[1, 4, 3]"
    ),
    "precision_at_3": (
        "Suggerisci 3 ID separati da virgola che l'utente apprezzerebbe con certezza.",
        "[2, 3, 4]"
    ),
    "coverage": (
        "Scegli esattamente 3 ID da 3 categorie diverse e rispondi separati da virgola.",
        "[2, 3, 6]"
    ),
}

def make_prompt(metric_key: str, metric_text: str, example: str = None):
    schema = ResponseSchema(
        name="ids",
        description="Lista JSON di soli ID numerici delle raccomandazioni, in ordine di priorità."
    )
    parser = StructuredOutputParser.from_response_schemas([schema])
    # prendo le vere istruzioni di formato dal parser e escapo le graffe
    fmt_instr = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

    # antepongo un esempio
    if example:
        fmt_instr = f"Esempio corretto: {example}\n" + fmt_instr

    # costruisco il template
    template_str = (
        f"Metric: {metric_key}\n"
        f"Descrizione: {metric_text}\n"
        f"Catalogo: {{catalog}}\n"
        f"{fmt_instr}"
    )
    return PromptTemplate(input_variables=["catalog"], template=template_str), parser

# ----------------------------
# 4. Costruzione delle catene
# ----------------------------
def build_prompt_chains():
    llm = ChatOpenAI(model=LLM_MODEL_ID, **COMMON_LLM_PARAMS)
    chains, parsers, raw_prompts = {}, {}, {}
    for metric, (text, example) in PROMPT_VARIANTS.items():
        prompt, parser = make_prompt(metric, text, example)
        chains[metric] = prompt | llm
        parsers[metric] = parser
        raw_prompts[metric] = prompt.template
    return chains, parsers, raw_prompts

def build_evaluator_chain():
    # STESSO modello LLM della generazione per la valutazione
    llm = ChatOpenAI(model=LLM_MODEL_ID, **COMMON_LLM_PARAMS)
 
    # Definisco lo schema di output desiderato per il valutatore
    final_ids_schema = ResponseSchema(
        name="final_ids",
        description="Lista JSON ottimizzata finale di massimo 3 ID numerici, basata sull'analisi di tutti i run."
    )
    justification_schema = ResponseSchema(
        name="justification",
        description="Breve spiegazione testuale del perché questi ID sono stati scelti come raccomandazione finale, considerando i risultati (output e metriche) dei diversi run."
    )
    parser = StructuredOutputParser.from_response_schemas([final_ids_schema, justification_schema])
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

    return prompt | llm, parser


# ----------------------------
# 5. Funzioni di utilità
# ----------------------------
def precision_at_k(recs, rel, k=3):
    return sum(1 for i in recs[:k] if i in rel) / k if k else 0

def coverage_metric(recs, all_ids):
    return len(set(recs)) / len(all_ids) if all_ids else 0

async def generate_for_metric(chain, parser, catalog_str):
    try:
        result = await chain.ainvoke({"catalog": catalog_str})
    except Exception as e:
        print(f"LLM error for {parser.name}: {e}")
        return [], ""
    content = getattr(result, "content", result)
    raw = "\n".join(content.splitlines()[:5]) + ("..." if len(content.splitlines()) > 5 else "")
    try:
        ids = parser.parse(content).get("ids", [])
    except Exception:
        try:
            ids = json.loads(content)
        except Exception:
            ids = []
    # Se ids è ancora stringa, lo carico come JSON
    if isinstance(ids, str):
        try:
            ids = json.loads(ids)
        except:
            ids = []
    # Cast a int e pulisco
    cleaned = []
    for x in ids:
        try:
            cleaned.append(int(x))
        except:
            pass
    return cleaned, raw

# ----------------------------
# 6. Main: generazione, metriche e salvataggio report
# ----------------------------
def main():
    chains, parsers, prompts = build_prompt_chains()
    # Costruisco il valutatore (ora uso Mistral e il nuovo prompt)
    evaluator_chain, evaluator_parser = build_evaluator_chain()

    all_ids = ALL_IDS
    num_runs = 3
    all_run_results = []

    print(f"Starting {num_runs} generation runs using {LLM_MODEL_ID}...")

    # CICLO ESTERNO PER I RUNS 
    for run_number in range(1, num_runs + 1):
        print(f"\n--- STARTING RUN {run_number}/{num_runs} ---")
        run_outputs = {}
        run_raw_outputs = {}
        run_metrics = {}

        # CICLO INTERNO PER LE METRICHE 
        print(f"  [Run {run_number}] Generating recommendations...")
        for metric in PROMPT_VARIANTS:
            # (stampa e chiamata a generate_for_metric come prima)
            ids, raw = asyncio.run(generate_for_metric(chains[metric], parsers[metric], CATALOG_STR))
            print(f"    Metric: {metric.upper()} -> Parsed IDs: {ids}")
            # print(f"    Raw: {raw}") # Opzionale: decommenta per vedere il raw
            run_outputs[metric] = ids
            run_raw_outputs[metric] = raw

        # CALCOLO METRICHE PER QUESTO RUN
        print(f"  [Run {run_number}] Calculating metrics...")
        for m, recs in run_outputs.items():
            run_metrics[m] = {
                "precision@3": precision_at_k(recs, all_ids),
                "coverage": coverage_metric(recs, all_ids)
            }
        print(f"    Metrics for Run {run_number}: {run_metrics}")

        # Salva i risultati completi di questo run
        all_run_results.append({
            "run_number": run_number,
            "outputs": run_outputs,
            "raw_outputs": run_raw_outputs, # Potrebbe essere utile per il debug
            "metrics": run_metrics
        })
        print(f"--- COMPLETED RUN {run_number}/{num_runs} ---")
    # FINE CICLO ESTERNO RUNS

    # VALUTAZIONE COMPLESSIVA POST-RUN
    print(f"\n--- STARTING OVERALL EVALUATION (using {LLM_MODEL_ID}) ---")
    final_recommendation_ids = []
    final_justification = "Evaluation could not be performed or parsed."

    if all_run_results: # Procedi solo se ci sono risultati dai run
        try:
            # Serializza tutti i risultati dei run in un JSON per il prompt
            all_run_results_json = json.dumps(all_run_results, ensure_ascii=False) # No indent for LLM

            # Invoca la catena del valutatore
            eval_res = asyncio.run(
                evaluator_chain.ainvoke({
                    "multi_run_results_json": all_run_results_json
                })
            )
            print("  Evaluator Raw Response:", eval_res.content) # Utile per debug

            # Tenta di parsare l'output strutturato (final_ids, justification)
            try:
                parsed_eval = evaluator_parser.parse(eval_res.content)
                raw_final_ids = parsed_eval.get("final_ids", [])
                final_justification = parsed_eval.get("justification", "No justification provided by evaluator.")

                # Pulisci e de-duplica gli ID finali
                # Gestisce il caso in cui l'LLM restituisca una stringa JSON invece di una lista
                if isinstance(raw_final_ids, str):
                   try:
                       raw_final_ids = json.loads(raw_final_ids)
                   except json.JSONDecodeError:
                       print("  Warning: final_ids from evaluator was a string but not valid JSON.")
                       raw_final_ids = []

                if isinstance(raw_final_ids, list):
                    seen = set()
                    uniq_final = []
                    for x in raw_final_ids:
                        try:
                            i = int(x)
                            if i not in seen:
                                seen.add(i)
                                uniq_final.append(i)
                        except (ValueError, TypeError):
                             print(f"  Warning: Could not convert item '{x}' to int in final_ids.")
                    final_recommendation_ids = uniq_final[:3] # Limita a 3 ID
                else:
                    print("  Warning: Parsed final_ids was not a list.")
                    final_recommendation_ids = []


            except Exception as parse_error:
                print(f"  Error parsing evaluator response: {parse_error}. Falling back.")
                final_justification = f"Failed to parse structured output. Raw: {eval_res.content}"
                # Fallback: tenta di estrarre almeno gli ID se il formato JSON è semplice
                try:
                    simple_json = json.loads(eval_res.content)
                    raw_final_ids = simple_json.get("final_ids", [])
                    if isinstance(raw_final_ids, str): raw_final_ids = json.loads(raw_final_ids) # controllo delle stringhe nidificate
                    if isinstance(raw_final_ids, list):
                      seen = set()
                      uniq_final = []
                      for x in raw_final_ids:
                           try:
                              i = int(x)
                              if i not in seen:
                                  seen.add(i)
                                  uniq_final.append(i)
                           except (ValueError, TypeError): pass
                      final_recommendation_ids = uniq_final[:3]
                    else: final_recommendation_ids = []
                except Exception: final_recommendation_ids = [] # Nessun ID recuperato

        except Exception as eval_error:
            print(f"  Error during evaluator LLM call: {eval_error}")
            # final_recommendation_ids resta []
            # final_justification resta il messaggio di errore predefinito

    print(f"  Final Recommended IDs: {final_recommendation_ids}")
    print(f"  Justification: {final_justification}")
    print("--- EVALUATION COMPLETE ---")

    # --- SALVATAGGIO REPORT FINALE ---
    report = {
        "generation_model": LLM_MODEL_ID,
        "evaluation_model": LLM_MODEL_ID, # Specifica il modello usato per valutare
        "num_runs": num_runs,
        "catalog_used": TOY_CATALOG, # Aggiungiamo il catalogo per riferimento
        "metric_prompts": prompts, # I prompt usati per la generazione
        "run_results": all_run_results, # Risultati dettagliati di ogni run
        "overall_recommendation": { # Sezione per la raccomandazione finale
            "ids": final_recommendation_ids,
            "justification": final_justification
        }
    }
    filename = f"report_multi_run_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nMulti-run report with evaluation saved to {filename}")

if __name__ == "__main__":
    main()