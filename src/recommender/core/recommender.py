"""
Sistema di raccomandazione multi-metrica basato su LangChain Agent.
"""

import os
import json
import asyncio
import pandas as pd
import re
import sys  # Aggiunto per sys.stdout.flush()
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.agents import initialize_agent, Tool, AgentType
from openai import RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Moduli locali
from src.recommender.utils.data_processor import (
    load_ratings, 
    load_movies, 
    filter_users_by_specific_users,
    create_user_profiles
)
from src.recommender.utils.rag_utils import MovieRAG
from src.recommender.core.metrics_utils import calculate_precision_at_k, calculate_coverage

# ----------------------------
# Setup ambiente e parametri
# ----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in environment")

COMMON_LLM_PARAMS = {
    "openai_api_base": "https://openrouter.ai/api/v1",
    "openai_api_key": OPENROUTER_API_KEY,
    "temperature": 0.7,
    "max_tokens": 512, # Potrebbe essere necessario aumentarlo per l'evaluator
}
LLM_MODEL_ID = "mistralai/mistral-large-2411"

# Setup LLM con retry
llm = ChatOpenAI(model=LLM_MODEL_ID, **COMMON_LLM_PARAMS)

@retry(
    reraise=True,
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError)
)
async def llm_arun_with_retry(prompt_str: str) -> str:
    response = await llm.ainvoke(prompt_str)
    # Gestisce sia stringa che AIMessage
    content = getattr(response, 'content', None)
    return content if content is not None else str(response)

# ----------------------------
# Definizioni Prompt e Parser
# ----------------------------
PROMPT_VARIANTS = {
    "precision_at_k": (
        "Sei un sistema di raccomandazione esperto che ottimizza per PRECISION@K. "
        "Il tuo obiettivo è raccomandare film che l'utente valuterà con un rating 4 o 5 su 5. "
        "Analizza attentamente il profilo dell'utente e concentrati sui seguenti elementi:\n"
        "1. Generi che l'utente ha costantemente valutato con punteggi elevati\n"
        "2. Attributi specifici (attori, registi, temi, epoche) presenti nei film apprezzati\n"
        "3. Modelli nelle valutazioni negative per evitare film simili\n\n"
        "La precision@k misura quanti dei film raccomandati saranno effettivamente valutati positivamente. "
        "Quando analizzi il catalogo, presta particolare attenzione a:\n"
        "- Corrispondenza di genere con i film valutati positivamente\n"
        "- Somiglianza tematica e stilistica ai film preferiti\n"
        "- Evita film simili a quelli che l'utente non ha apprezzato\n\n"
        "NON raccomandare film in base alla popolarità generale o alle tendenze, a meno che queste "
        "caratteristiche non si allineino alle preferenze specifiche di questo utente. "
        "Fornisci ESATTAMENTE 3 film che l'utente molto probabilmente valuterà positivamente."
    ),
    "coverage": (
        "Sei un sistema di raccomandazione esperto che ottimizza per COVERAGE. "
        "Data una lista di film, consiglia 3 film che massimizzano la copertura di diversi generi cinematografici, "
        "MA che siano comunque rilevanti per le preferenze specifiche dell'utente di cui stai analizzando il profilo. "
        "La coverage misura la proporzione dell'intero catalogo che il sistema è in grado di raccomandare. "
        "L'obiettivo è esplorare meglio lo spazio dei film disponibili e ridurre il rischio di filter bubble. "
        "Assicurati che le tue raccomandazioni coprano generi diversi tra loro, ma che siano allineati con i gusti dell'utente. "
        "IMPORTANTE: Fai riferimento specifico ai film che l'utente ha apprezzato per scoprire generi correlati ma diversi. "
        "Ogni utente deve ricevere raccomandazioni personalizzate in base al suo profilo unico. "
        "NON raccomandare più di 3 film."
    )
}

response_schema = ResponseSchema(
    name="recommendations",
    description="Lista di esattamente 3 ID di film raccomandati, in formato JSON. Es: [145, 270, 381]"
)
explanation_schema = ResponseSchema(
    name="explanation",
    description="Breve spiegazione del motivo per cui hai scelto questi film in base alla metrica richiesta."
)
parser = StructuredOutputParser.from_response_schemas([response_schema, explanation_schema])
FORMAT_INSTRUCTIONS = parser.get_format_instructions()

def create_metric_prompt(metric_name: str, metric_description: str) -> PromptTemplate:
    """Crea un PromptTemplate per una specifica metrica."""
    return PromptTemplate(
        input_variables=["catalog", "user_profile"],
        template=(
            f"# Compito: Raccomandatore di film ottimizzato per {metric_name.upper()}\n\n"
            f"{metric_description}\n\n"
            "# Profilo utente:\n"
            "{user_profile}\n\n"
            "# Catalogo film:\n"
            "{catalog}\n\n"
            "# Formato di output richiesto:\n"
            "La tua risposta deve contenere ESATTAMENTE le chiavi 'recommendations' e 'explanation' in formato JSON.\n"
            "Esempio di formato JSON corretto:\n"
            "```json\n"
            "{{\n"
            "  \"recommendations\": [145, 270, 381],\n"
            "  \"explanation\": \"Spiegazione concisa della tua scelta basata sulla metrica specifica...\"\n"
            "}}\n"
            "```\n"
            "Assicurati che 'recommendations' sia una lista di 3 ID numerici di film e 'explanation' una stringa.\n"
            "NON includere nient'altro nella risposta oltre al blocco JSON."
        )
    )

class RecommenderSystem:
    """
    Sistema di raccomandazione unificato basato su Agent e Tool.
    """
    
    def __init__(self, specific_user_ids: List[int] = [1, 2], model_id: str = LLM_MODEL_ID):
        self.specific_user_ids = specific_user_ids
        self.model_id = model_id
        self.llm = llm
        self.filtered_ratings = None
        self.user_profiles = None 
        self.movies = None
        self.rag = None
        self.agent = None
        self.metric_tools = []
        self.evaluator_tool = None
        self.datasets_loaded = False
        self.current_prompt_variants = PROMPT_VARIANTS.copy() # Inizializza con default

    def _load_datasets(self, force_reload: bool = False) -> None:
        """Carica e prepara i dataset internamente."""
        if not self.datasets_loaded or force_reload:
            print("\n=== Caricamento e processamento dei dataset ===")
            try:
                processed_dir = os.path.join('data', 'processed')
                os.makedirs(processed_dir, exist_ok=True)
                ratings_file = os.path.join(processed_dir, 'filtered_ratings_specific.csv')
                profiles_file = os.path.join(processed_dir, 'user_profiles_specific.csv')
                movies_file = os.path.join(processed_dir, 'movies.csv')

                if not force_reload and all(os.path.exists(f) for f in [ratings_file, profiles_file, movies_file]):
                    print("Caricamento dati da file elaborati...")
                    self.filtered_ratings = pd.read_csv(ratings_file)
                    self.user_profiles = pd.read_csv(profiles_file, index_col=0) 
                    if self.user_profiles is not None:
                         self.user_profiles.index.name = 'user_id'
                    self.movies = pd.read_csv(movies_file)
                else:
                    print("Elaborazione dati dal dataset grezzo...")
                    ratings = load_ratings()
                    self.movies = load_movies()
                    self.filtered_ratings = filter_users_by_specific_users(ratings, self.specific_user_ids)
                    self.user_profiles = create_user_profiles(self.filtered_ratings)
                    self.filtered_ratings.to_csv(ratings_file, index=False)
                    self.user_profiles.to_csv(profiles_file, index=True) 
                    self.movies.to_csv(movies_file, index=False)
                
                if self.user_profiles is None or self.user_profiles.empty:
                     print("ATTENZIONE: user_profiles vuoto dopo il caricamento!")
                
                print(f"Dataset processato: {len(self.movies)} film, {len(self.user_profiles) if self.user_profiles is not None else 0} profili (ID: {self.specific_user_ids}).")
                self.datasets_loaded = True
            except Exception as e:
                print(f"Errore caricamento dataset: {e}")
                import traceback; traceback.print_exc()
                raise
        else:
            print("Dataset già caricati.")

    def _initialize_rag(self, force_recreate_vector_store: bool = False) -> None:
        """Inizializza il sistema RAG."""
        if not self.datasets_loaded:
            self._load_datasets()
        if self.movies is None or self.movies.empty:
             print("Movies non caricati, impossibile inizializzare RAG."); return
        print("\n=== Preparazione RAG ===")
        self.rag = MovieRAG()
        movies_list = self.movies.to_dict('records')
        self.rag.load_or_create_vector_store(movies_list, force_recreate=force_recreate_vector_store)
        try:
            catalog_json = self.rag.get_optimized_catalog_for_llm(movies_list)
            catalog_path = os.path.join('data', 'processed', 'optimized_catalog.json')
            os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
            with open(catalog_path, 'w', encoding='utf-8') as f: f.write(catalog_json)
            print(f"Catalogo ottimizzato RAG salvato.")
        except Exception as e: print(f"Attenzione: impossibile generare/salvare catalogo ottimizzato RAG: {e}")

    def _parse_tool_response(self, response_content: str, metric_name: str) -> Dict:
        """Estrae JSON strutturato dalla risposta del tool."""
        try:
            # 1. Cerca blocco JSON completo
            json_match = re.search(r'({[\s\S]*?"recommendations"[\s\S]*?"explanation"[\s\S]*?})', response_content)
            if json_match:
                json_str = json_match.group(1)
                json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                json_str = re.sub(r'```json|```', '', json_str).strip()
                try: 
                    data = json.loads(json_str)
                    # Validazione minima
                    if isinstance(data.get('recommendations'), list) and isinstance(data.get('explanation'), str):
                         return {"metric": metric_name, "recommendations": data['recommendations'], "explanation": data['explanation']}
                except json.JSONDecodeError: pass # Prova altri metodi
            
            # 2. Cerca chiavi separate
            recs_match = re.search(r'"recommendations":\s*(\[[\s\d,]*\])', response_content)
            exp_match = re.search(r'"explanation":\s*"([^"]*)"', response_content)
            if recs_match:
                try: recs = json.loads(recs_match.group(1))
                except: recs = []
                explanation = exp_match.group(1) if exp_match else "Explanation extracted partially"
                return {"metric": metric_name, "recommendations": recs, "explanation": explanation}

            # 3. Cerca solo lista di ID
            rec_match_list = re.search(r'(\[\s*\d+\s*(?:,\s*\d+\s*)*\])', response_content) # Regex migliorata
            if rec_match_list:
                 try: recs = json.loads(rec_match_list.group(1))
                 except: recs = []
                 return {"metric": metric_name, "recommendations": recs, "explanation": f"List extracted: {recs}"}

            # 4. Estrai numeri
            numbers = re.findall(r'\b\d+\b', response_content)
            if numbers:
                unique_nums = list(dict.fromkeys([int(n) for n in numbers]))
                if len(unique_nums) >= 3:
                    return {"metric": metric_name, "recommendations": unique_nums[:3], "explanation": f"Numbers extracted: {unique_nums[:3]}"}
            
            print(f"Impossibile fare il parsing della risposta per {metric_name}")
            raise ValueError("Cannot extract recommendations from the response.")
        except Exception as e:
            print(f"Errore parsing JSON per {metric_name}: {e}")
            return {"metric": metric_name, "recommendations": [], "explanation": f"Parsing Error: {e}"}

    def _build_metric_tools(self) -> List[Tool]:
        """Costruisce i Tools per ciascuna metrica di raccomandazione."""
        print("Costruzione metric tools...")
        metric_tools = []
        prompt_variants_to_use = self.current_prompt_variants # Usa le varianti correnti

        for metric_name, metric_desc in prompt_variants_to_use.items():
            prompt_template = create_metric_prompt(metric_name, metric_desc)
            
            async def run_metric_agent_tool(catalog: str, user_profile: str, _metric=metric_name, _prompt=prompt_template):
                try:
                    prompt_str = _prompt.format(catalog=catalog, user_profile=user_profile)
                    response_content = await llm_arun_with_retry(prompt_str)
                    print(f"Raw response from {_metric} tool: {response_content[:300]}...")
                    return self._parse_tool_response(response_content, _metric)
                except Exception as e:
                    print(f"Error running {_metric} tool: {e}")
                    return {"metric": _metric, "recommendations": [], "explanation": f"Execution Error: {e}"}
            
            metric_tools.append(
                Tool(
                    name=f"recommender_{metric_name}",
                    func=None,
                    description=f"Genera raccomandazioni ottimizzate per {metric_name}. Input: catalog (JSON string), user_profile (JSON string)",
                    coroutine=run_metric_agent_tool
                )
            )
        self.metric_tools = metric_tools
        return metric_tools

    def _build_evaluator_tool(self) -> Tool:
        """Costruisce il Tool per valutare e combinare le raccomandazioni."""
        print("Costruzione evaluator tool...")
        eval_prompt = PromptTemplate(
            input_variables=["all_recommendations", "catalog"],
             template=(
                 "# Compito: Valutazione e combinazione di raccomandazioni multi-metrica\n\n"
                 "Sei un sistema avanzato che deve combinare raccomandazioni di film generate da diversi sistemi specializzati.\n"
                 "Ogni sistema si è concentrato su una metrica diversa (es. precision@k, coverage).\n\n"
                 "# Raccomandazioni Ricevute per Utente (JSON string):\n"
                 "{all_recommendations}\n\n"
                 "# Catalogo Film (JSON string):\n"
                 "{catalog}\n\n"
                 "# Istruzioni:\n"
                 "1. Analizza attentamente le raccomandazioni per ogni metrica e per ogni utente.\n"
                 "2. Considera i punti di forza di ciascuna metrica (precision@k = rilevanza, coverage = diversità).\n"
                 "3. Crea una lista finale OTTIMALE e UNICA di 3 film che bilanci le diverse metriche in modo sensato per un ipotetico utente medio rappresentato dai dati forniti.\n"
                 "4. Spiega DETTAGLIATAMENTE la tua logica di selezione e i trade-off considerati per arrivare alla lista finale aggregata.\n\n"
                 "# Formato di output richiesto (JSON ESATTO):\n"
                 "```json\n"
                 "{{\n"
                 "  \"final_recommendations\": [ID_FILM_1, ID_FILM_2, ID_FILM_3],\n"
                 "  \"justification\": \"Spiegazione dettagliata del perché hai scelto questi 3 film specifici come bilanciamento ottimale...\",\n"
                 "  \"trade_offs\": \"Descrizione dei trade-off considerati (es. sacrificato un po' di precisione per più copertura, scelto film popolari vs. nicchia...)\"\n"
                 "}}\n"
                 "```\n"
                 "IMPORTANTE: Rispetta ESATTAMENTE il formato JSON. 'final_recommendations' deve contenere 3 ID numerici. Le spiegazioni devono essere stringhe."
            )
        )
        
        async def evaluate_recommendations_tool(all_recommendations_str: str, catalog_str: str) -> Dict:
            max_retries = 3 
            retry_delay = 2 
            for attempt in range(max_retries):
                try:
                    print(f"\nAttempt {attempt+1}/{max_retries} to evaluate recommendations...")
                    prompt_str = eval_prompt.format(all_recommendations=all_recommendations_str, catalog=catalog_str)
                    response_content = await llm_arun_with_retry(prompt_str)
                    print(f"Raw evaluator response: {response_content[:300]}...")
                    
                    # Estrazione JSON robusta per l'evaluator
                    json_match = re.search(r'({[\s\S]*?"final_recommendations"[\s\S]*?"justification"[\s\S]*?"trade_offs"[\s\S]*?})', response_content)
                    if json_match:
                        json_str = json_match.group(1)
                        json_str = re.sub(r'[\n\r\t]', ' ', json_str).strip()
                        json_str = re.sub(r'```json|```', '', json_str).strip()
                        try:
                            data = json.loads(json_str)
                            if isinstance(data.get("final_recommendations"), list) and \
                               len(data.get("final_recommendations")) == 3 and \
                               isinstance(data.get("justification"), str) and \
                               isinstance(data.get("trade_offs"), str):
                                return {
                                    "final_recommendations": data["final_recommendations"],
                                    "justification": data["justification"],
                                    "trade_offs": data["trade_offs"]
                                }
                            else: print("Evaluator JSON validation failed.")
                        except json.JSONDecodeError: print(f"Malformed JSON from evaluator: {json_str[:100]}...")
                    
                    print(f"Failed to parse evaluator response on attempt {attempt+1}.")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay); retry_delay *= 2
                    else:
                        raise ValueError("Could not parse evaluator response after multiple retries.")
                except Exception as inner_e:
                    print(f"Error during evaluator tool execution attempt {attempt+1}: {inner_e}")
                    if attempt < max_retries - 1:
                         await asyncio.sleep(retry_delay); retry_delay *= 2
                    else:
                        return {"final_recommendations": [], "justification": f"Evaluation failed: {inner_e}", "trade_offs": "N/A"}
            return {"final_recommendations": [], "justification": "Evaluation failed unexpectedly.", "trade_offs": "N/A"}

        self.evaluator_tool = Tool(
            name="evaluate_recommendations",
            func=None,
            description="Valuta e combina raccomandazioni generate. Input: all_recommendations (JSON string), catalog (JSON string)",
            coroutine=evaluate_recommendations_tool
        )
        return self.evaluator_tool
        
    def _initialize_agent(self) -> None:
        """Inizializza l'Agent di LangChain con i tool costruiti."""
        print("Inizializzazione Agent...")
        if not self.metric_tools: self._build_metric_tools()
        if not self.evaluator_tool: self._build_evaluator_tool()
        all_tools = self.metric_tools + [self.evaluator_tool]
        try:
            if not all(hasattr(tool, 'coroutine') and callable(tool.coroutine) for tool in all_tools):
                raise ValueError("Tutti i tools devono avere una coroutine.")
            self.agent = initialize_agent(
                all_tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                verbose=True, handle_parsing_errors=True
            )
            print("LangChain Agent inizializzato.")
        except Exception as e:
             print(f"Errore inizializzazione Agent: {e}")
             import traceback; traceback.print_exc()
             self.agent = None

    def initialize_system(self, force_reload_data: bool = False, force_recreate_vector_store: bool = False) -> None:
        """Metodo pubblico per inizializzare o reinizializzare il sistema."""
        print("\n=== Inizializzazione Sistema ===")
        self._load_datasets(force_reload=force_reload_data)
        self._initialize_rag(force_recreate_vector_store=force_recreate_vector_store)
        self._initialize_agent()
        print("=== Sistema Inizializzato ===")

    def get_optimized_catalog(self, limit: int = 100) -> str:
        """Ottiene il catalogo ottimizzato per l'LLM."""
        catalog_path = os.path.join('data', 'processed', 'optimized_catalog.json')
        try:
            if os.path.exists(catalog_path):
                with open(catalog_path, 'r', encoding='utf-8') as f: catalog_data = json.load(f)
                return json.dumps(catalog_data[:limit] if limit else catalog_data, ensure_ascii=False)
            elif self.rag:
                 print("Catalogo ottimizzato non trovato, genero da RAG...")
                 if self.movies is None or self.movies.empty: self._load_datasets()
                 movies_list = self.movies.to_dict('records')
                 return self.rag.get_optimized_catalog_for_llm(movies_list, limit=limit)
            else: print("Attenzione: RAG non inizializzato."); return "[]"
        except Exception as e: print(f"Errore get_optimized_catalog: {e}"); return "[]"
             
    async def run_recommendation_pipeline(self, use_prompt_variants: Dict = None) -> Tuple[Dict, Dict]:
        """
        Esegue l'intera pipeline di raccomandazione per tutti gli utenti specificati.
        Invoca i tool delle metriche e poi il tool di valutazione.
        """
        if not self.agent: raise RuntimeError("Agent non inizializzato. Chiamare initialize_system() prima.")
        if not self.datasets_loaded or self.user_profiles is None or self.user_profiles.empty:
             raise RuntimeError("Dataset non caricati o profili utente vuoti. Chiamare initialize_system() prima.")

        # Imposta le varianti di prompt da usare per questa run
        self.current_prompt_variants = use_prompt_variants if use_prompt_variants is not None else PROMPT_VARIANTS
        # Ricostruisce i tool metriche con le varianti correnti
        current_metric_tools = self._build_metric_tools() 
        # Nota: l'agent NON viene reinizializzato qui, usa i tool passati al momento della chiamata (se usassimo agent.arun)

        user_metric_results = {}
        if self.user_profiles is None: raise RuntimeError("user_profiles non inizializzato.")

        for user_id in self.specific_user_ids:
            if user_id not in self.user_profiles.index:
                print(f"Attenzione: Utente {user_id} non trovato."); continue

            profile_series = self.user_profiles.loc[user_id]
            def safe_load_list(item):
                 if isinstance(item, list): return item
                 if pd.isna(item): return []
                 if isinstance(item, str):
                     try: return json.loads(item.replace("'", '"')) if isinstance(json.loads(item.replace("'", '"')), list) else []
                     except: 
                         if item.startswith('[') and item.endswith(']'): 
                             try: return [int(x.strip()) for x in item[1:-1].split(',') if x.strip().isdigit()]
                             except: pass
                 return []
            liked = safe_load_list(profile_series.get("liked_movies"))
            disliked = safe_load_list(profile_series.get("disliked_movies"))
            profile_summary = json.dumps({"user_id": int(user_id),"liked_movies": liked,"disliked_movies": disliked}, ensure_ascii=False)

            print(f"\n--- Raccomandazioni per utente {user_id} ---")
            catalog_json = "[]"
            try:
                if self.rag:
                    cat_p = self.rag.similarity_search(profile_summary, k=100, metric_focus="precision_at_k", user_id=int(user_id))
                    cat_c = self.rag.similarity_search("diversi generi " + profile_summary, k=100, metric_focus="coverage", user_id=int(user_id))
                    merged = self.rag.merge_catalogs(cat_p, cat_c)
                    catalog_json = json.dumps(merged[:100], ensure_ascii=False)
                else: catalog_json = self.get_optimized_catalog(limit=100)
            except Exception as e: print(f"Errore RAG user {user_id}: {e}. Uso catalogo generico."); catalog_json = self.get_optimized_catalog(limit=100)
            
            # Esegui i tool delle metriche direttamente
            results_for_user = {}
            tasks = [tool.coroutine(catalog=catalog_json, user_profile=profile_summary) for tool in current_metric_tools if tool.coroutine]
            tool_names = [tool.name.replace("recommender_", "") for tool in current_metric_tools if tool.coroutine]
            
            if tasks:
                metric_outputs = await asyncio.gather(*tasks, return_exceptions=True)
                for i, output in enumerate(metric_outputs):
                    metric_name = tool_names[i]
                    if isinstance(output, Exception): results_for_user[metric_name] = {"metric": metric_name, "recommendations": [], "explanation": f"Error: {output}"}
                    elif isinstance(output, dict): results_for_user[metric_name] = output
                    else: results_for_user[metric_name] = {"metric": metric_name, "recommendations": [], "explanation": f"Unexpected type: {type(output)}"}
            user_metric_results[user_id] = results_for_user

        # Valutazione aggregata
        print("\n--- Valutazione Aggregata ---")
        final_evaluation = {}
        if user_metric_results:
            try:
                all_results_str = json.dumps(user_metric_results, ensure_ascii=False, indent=2)
                eval_catalog = self.get_optimized_catalog(limit=100)
                if self.evaluator_tool and self.evaluator_tool.coroutine:
                    final_evaluation = await self.evaluator_tool.coroutine(all_recommendations_str=all_results_str, catalog_str=eval_catalog)
                else: raise ValueError("Evaluator tool/coroutine non definito.")
                print(f"Final recommendations: {final_evaluation.get('final_recommendations', [])}")
            except Exception as e:
                 print(f"Errore evaluator tool: {e}"); import traceback; traceback.print_exc()
                 final_evaluation = {"final_recommendations": [], "justification": f"Evaluation Error: {e}", "trade_offs": "N/A"}
        else: final_evaluation = {"final_recommendations": [], "justification": "No results to evaluate.", "trade_offs": "N/A"}

        # Ripristina le varianti di prompt di default se necessario
        self.current_prompt_variants = PROMPT_VARIANTS.copy() 
        self._build_metric_tools() # Ricostruisce i tool con i prompt di default

        return user_metric_results, final_evaluation

    def save_results(self, metric_results: Dict, final_evaluation: Dict, metrics_calculated: Dict = None):
        """Salva i risultati della pipeline su file."""
        # Usa un blocco per garantire che il file sia chiuso prima di procedere
        try:
            with open("metric_recommendations.json", "w", encoding="utf-8") as f:
                json.dump(metric_results, f, ensure_ascii=False, indent=2)
                # Flush esplicito per garantire che il file sia scritto completamente
                f.flush()
                os.fsync(f.fileno())
            print("\nRisultati intermedi salvati: metric_recommendations.json")
            sys.stdout.flush()  # Flush dello stdout
        except Exception as e: 
            print(f"Errore salvataggio metric_recommendations.json: {e}")
            sys.stdout.flush()
        
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "metric_recommendations": metric_results,
            "final_evaluation": final_evaluation
        }
        if metrics_calculated: 
            result_data["metrics"] = metrics_calculated
        
        try:
            with open("recommendation_results.json", "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                # Flush esplicito per garantire che il file sia scritto completamente
                f.flush()
                os.fsync(f.fileno())
            print("Risultati finali salvati: recommendation_results.json")
            sys.stdout.flush()  # Flush dello stdout
        except Exception as e: 
            print(f"Errore salvataggio recommendation_results.json: {e}")
            sys.stdout.flush()

    def calculate_and_display_metrics(self, metric_results: Dict, final_evaluation: Dict) -> Dict:
        """Calcola e visualizza le metriche."""
        if not self.datasets_loaded or self.movies is None or self.filtered_ratings is None:
             print("Dataset non caricati, metriche non calcolabili."); return {}
        try:
            prec_recs, cov_recs = [], []
            for uid, u_metrics in metric_results.items():
                prec_recs.extend(u_metrics.get('precision_at_k', {}).get('recommendations', []))
                cov_recs.extend(u_metrics.get('coverage', {}).get('recommendations', []))
            prec_recs, cov_recs = list(dict.fromkeys(prec_recs)), list(dict.fromkeys(cov_recs))
            final_recs = final_evaluation.get('final_recommendations', [])
            
            # 2. Prepara dati per metriche
            all_movie_ids = self.movies['movie_id'].tolist()
            
            # --- Inizio Logica Vecchia/Corretta per relevant_items ---
            # Determiniamo i film rilevanti come quelli con rating medio >=4 (dal dataset completo)
            ratings_path = os.path.join('data', 'raw', 'ratings.dat')
            relevant_items = [] # Inizializza lista vuota
            try:
                print("Tentativo di caricare tutti i rating per calcolare la rilevanza globale...")
                # Carica solo le colonne necessarie dal file completo
                all_ratings = pd.read_csv(
                    ratings_path, 
                    sep='::', 
                    engine='python', 
                    header=None, 
                    names=['user_id','movie_id','rating','timestamp'],
                    usecols=['movie_id', 'rating'] # Carica solo colonne utili
                )
                avg_ratings = all_ratings.groupby('movie_id')['rating'].mean()
                relevant_items = avg_ratings[avg_ratings >= 4].index.tolist()
                print(f"Trovati {len(relevant_items)} film rilevanti globalmente (avg_rating >= 4).")
            except FileNotFoundError:
                print(f"Attenzione: File {ratings_path} non trovato. Uso fallback basato su dati filtrati.")
            except Exception as e:
                print(f"Errore nel caricamento/calcolo dei rating globali: {e}. Uso fallback.")
            
            # Fallback: se il caricamento fallisce o non trova item rilevanti, 
            # usa i film che hanno almeno un rating >=4 nell'insieme filtrato
            if not relevant_items:
                 print("Fallback: Calcolo rilevanza basata solo sui dati filtrati (rating >= 4 da utenti specifici).")
                 if self.filtered_ratings is not None:
                      relevant_items = self.filtered_ratings[self.filtered_ratings['rating'] >= 4]['movie_id'].unique().tolist()
                 else:
                      relevant_items = [] # Nessun dato filtrato disponibile
                 print(f"Trovati {len(relevant_items)} film rilevanti nel fallback.")
            # --- Fine Logica Vecchia/Corretta per relevant_items ---

            # Rimuovo la vecchia riga che usava solo i dati filtrati:
            # relevant = self.filtered_ratings[self.filtered_ratings['rating'] >= 4]['movie_id'].unique().tolist()
            
            if not relevant_items: 
                 print("Attenzione: nessun film rilevante trovato (rating>=4). Precision@k sarà 0.")

            # 3. Calcola precision@k
            # Usa la lista 'relevant_items' calcolata sopra
            precision_pak_value = calculate_precision_at_k(prec_recs, relevant_items)
            coverage_pak_value = calculate_precision_at_k(cov_recs, relevant_items)
            final_pak_value = calculate_precision_at_k(final_recs, relevant_items)
            
            all_genres = set(g for genres in self.movies['genres'].dropna() for g in genres.split('|'))
            n_genres = len(all_genres)
            genre_cov = {}
            def get_genres(mid): 
                m = self.movies[self.movies['movie_id'] == mid]
                return set(m.iloc[0]['genres'].split('|')) if not m.empty and pd.notna(m.iloc[0]['genres']) else set()
            for name, recs in [("precision_at_k", prec_recs), ("coverage", cov_recs), ("final", final_recs)]:
                g = set().union(*[get_genres(mid) for mid in recs])
                genre_cov[name] = len(g) / n_genres if n_genres > 0 else 0
            total_cov = len(set(prec_recs + cov_recs + final_recs)) / len(all_movie_ids) if all_movie_ids else 0
            
            print("\nMetriche Calcolate:")
            print(f"  Precision@k (prec): {precision_pak_value:.4f}, (cov): {coverage_pak_value:.4f}, (final): {final_pak_value:.4f}")
            print(f"  Genre Coverage (prec): {genre_cov.get('precision_at_k', 0):.4f}, (cov): {genre_cov.get('coverage', 0):.4f}, (final): {genre_cov.get('final', 0):.4f}")
            print(f"  Total Coverage (items): {total_cov:.4f}")
            
            metrics = {
                "precision_at_k": {"precision_score": precision_pak_value, "genre_coverage": genre_cov.get('precision_at_k', 0)},
                "coverage": {"precision_score": coverage_pak_value, "genre_coverage": genre_cov.get('coverage', 0)},
                "final_recommendations": {"precision_score": final_pak_value, "genre_coverage": genre_cov.get('final', 0)},
                "total_coverage": total_cov
            }
            return metrics
        except Exception as e: print(f"Errore calcolo metriche: {e}"); import traceback; traceback.print_exc(); return {"error": str(e)}

    # ----- Metodi per esperimenti ----- 
    async def generate_recommendations_with_custom_prompt(self, prompt_variants: Dict, experiment_name: str ="custom_experiment") -> Tuple[Dict, str]:
        print(f"\n=== Esecuzione Esperimento: {experiment_name} ===")
        metric_results, final_evaluation = await self.run_recommendation_pipeline(use_prompt_variants=prompt_variants)
        metrics = self.calculate_and_display_metrics(metric_results, final_evaluation)
        os.makedirs("experiments", exist_ok=True)
        filename = f"experiments/experiment_{experiment_name}.json"
        result = {
            "timestamp": datetime.now().isoformat(),
            "experiment_info": {"name": experiment_name, "prompt_variants": prompt_variants},
            "metric_recommendations": metric_results,
            "final_evaluation": final_evaluation,
            "metrics": metrics
        }
        try:
            with open(filename, "w", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Risultati esperimento salvati: {filename}")
        except Exception as e: print(f"Errore salvataggio file esperimento {filename}: {e}")
        return result, filename
        
    async def generate_standard_recommendations(self) -> Dict:
        """Genera raccomandazioni standard, calcola metriche e salva."""
        print("\n=== Esecuzione Pipeline Standard ===")
        metric_results, final_evaluation = await self.run_recommendation_pipeline()
        metrics = self.calculate_and_display_metrics(metric_results, final_evaluation)
        self.save_results(metric_results, final_evaluation, metrics_calculated=metrics)
        
        # Dà tempo all'event loop di stabilizzarsi prima di stampare i messaggi finali
        await asyncio.sleep(0.1)
        sys.stdout.flush()
        
        print("\n=== Standard Recommendation Process Complete ===")
        print(f"Final recommendations: {final_evaluation.get('final_recommendations', [])}")
        sys.stdout.flush()
        
        return {"timestamp": datetime.now().isoformat(), "metric_recommendations": metric_results, "final_evaluation": final_evaluation, "metrics": metrics} 