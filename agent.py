# ----------------------------
# Import e setup iniziale
# ----------------------------
import os
import json
import asyncio
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ----------------------------
# LangChain imports
# ----------------------------
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.agents import initialize_agent, Tool, AgentType
from openai import RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ----------------------------
# Import dei moduli locali
# ----------------------------
from src.recommender.utils.data_processor import process_dataset, get_movie_catalog_for_llm, filter_users_by_specific_users, load_ratings, load_movies, create_user_profiles
from src.recommender.utils.rag_utils import MovieRAG, calculate_precision_at_k, calculate_coverage

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
LLM_MODEL_ID = "openai/gpt-4o-mini"

# ----------------------------
# 2. Caricamento del dataset
# ----------------------------
DATASETS_LOADED = False

def load_datasets(force_reload=False):
    """Carica e prepara i dataset"""
    global DATASETS_LOADED
    global filtered_ratings, user_profiles, movies  # Aggiunte variabili globali per accesso
    
    if not DATASETS_LOADED or force_reload:
        print("\n=== Caricamento e processamento dei dataset ===\n")
        try:
            # NUOVA IMPLEMENTAZIONE: Filtra solo utenti con ID 1 e 2
            # Verifica se i file elaborati esistono
            processed_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
            if not force_reload and all(os.path.exists(os.path.join(processed_dir, f)) 
                                   for f in ['filtered_ratings_specific.csv', 'user_profiles_specific.csv', 'movies.csv']):
                print("Caricamento dati da file elaborati con utenti specifici...")
                filtered_ratings = pd.read_csv(os.path.join(processed_dir, 'filtered_ratings_specific.csv'))
                user_profiles = pd.read_csv(os.path.join(processed_dir, 'user_profiles_specific.csv'), index_col=0)
                movies = pd.read_csv(os.path.join(processed_dir, 'movies.csv'))
            else:
                print("Elaborazione dati dal dataset grezzo...")
                # Carica i dati
                ratings = load_ratings()
                movies = load_movies()
                
                # Filtra solo utenti con ID 1 e 2
                filtered_ratings = filter_users_by_specific_users(ratings, [1, 2])
                
                # Crea profili utente
                user_profiles = create_user_profiles(filtered_ratings)
                
                # Salva i dati elaborati
                os.makedirs(processed_dir, exist_ok=True)
                filtered_ratings.to_csv(os.path.join(processed_dir, 'filtered_ratings_specific.csv'), index=False)
                user_profiles.to_csv(os.path.join(processed_dir, 'user_profiles_specific.csv'))
                movies.to_csv(os.path.join(processed_dir, 'movies.csv'), index=False)
            
            # IMPLEMENTAZIONE ORIGINALE (commentata)
            # # Processa il dataset
            # filtered_ratings, user_profiles, movies = process_dataset()
            # print(f"Dataset processato con successo. {len(movies)} film, {len(user_profiles)} profili utente con almeno 100 valutazioni.")
            
            print(f"Dataset processato con successo. {len(movies)} film, {len(user_profiles)} profili utente specifici (ID: 1, 2).")
            DATASETS_LOADED = True
            
            # Prepara il catalogo ottimizzato per il LLM
            print("\n=== Preparazione del catalogo ottimizzato per RAG ===\n")
            rag = MovieRAG(model_name=LLM_MODEL_ID)
            
            # Converti movies DataFrame in lista di dizionari
            movies_list = movies.to_dict('records')
            
            # Inizializza il vector store senza chiamare initialize_embeddings che non esiste
            rag.load_or_create_vector_store(movies_list, force_recreate=force_reload)
            
            # Genera e salva il catalogo ottimizzato
            catalog_json = rag.get_optimized_catalog_for_llm(movies_list)
            
            # Salva il catalogo ottimizzato
            catalog_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'optimized_catalog.json')
            with open(catalog_path, 'w', encoding='utf-8') as f:
                f.write(catalog_json)
                
            print(f"Catalogo ottimizzato salvato in {catalog_path}")
            
            return filtered_ratings, user_profiles, movies
            
        except Exception as e:
            print(f"Errore durante il caricamento dei dataset: {e}")
            raise
    else:
        print("Dataset già caricati.")
        return filtered_ratings, user_profiles, movies  # Restituisce le variabili globali

# ----------------------------
# 3. Ottenimento del catalogo ottimizzato
# ----------------------------
def get_optimized_catalog(limit=50):
    """Ottiene il catalogo ottimizzato per l'LLM"""
    catalog_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'optimized_catalog.json')
    
    if os.path.exists(catalog_path):
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog_json = f.read()
            
        # Limita il numero di film se necessario
        if limit:
            catalog = json.loads(catalog_json)
            catalog = catalog[:limit]
            return json.dumps(catalog, ensure_ascii=False)
        
        return catalog_json
    else:
        print("Catalogo ottimizzato non trovato. Eseguendo il processamento dei dataset...")
        load_datasets(force_reload=True)
        return get_optimized_catalog(limit)

# ----------------------------
# 4. Metric-driven prompts
# ----------------------------
PROMPT_VARIANTS = {
    "precision_at_k": (
        "Sei un sistema di raccomandazione esperto che ottimizza per PRECISION@K. "
        "Data una lista di film, consiglia i 3 film più rilevanti per un utente generico. "
        "Massimizza la proporzione di film raccomandati che saranno effettivamente apprezzati. "
        "La precision@k misura la frazione di film raccomandati che l'utente valuterebbe positivamente. "
        "Concentrati sui film più popolari e di alta qualità. "
        "NON raccomandare più di 3 film."
    ),
    "coverage": (
        "Sei un sistema di raccomandazione esperto che ottimizza per COVERAGE. "
        "Data una lista di film, consiglia 3 film che massimizzano la copertura di diversi generi cinematografici. "
        "La coverage misura la proporzione dell'intero catalogo che il sistema è in grado di raccomandare. "
        "L'obiettivo è esplorare meglio lo spazio dei film disponibili e ridurre il rischio di filter bubble. "
        "Assicurati che le tue raccomandazioni coprano generi diversi tra loro. "
        "NON raccomandare più di 3 film."
    )
}

# ----------------------------
# 5. Response schema e parser
# ----------------------------
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

# ----------------------------
# 6. Funzioni di utilità per il prompt
# ----------------------------
def create_metric_prompt(metric_name: str, metric_description: str):
    return PromptTemplate(
        input_variables=["catalog"],
        template=(
            f"# Compito: Raccomandatore di film ottimizzato per {metric_name.upper()}\n\n"
            f"{metric_description}\n\n"
            "# Catalogo film:\n"
            "{catalog}\n\n"
            "# Formato di output richiesto:\n"
            "La tua risposta deve contenere esattamente queste chiavi in questo formato:\n"
            "```json\n"
            "{{\n"
            '  "recommendations": [145, 270, 381],\n'
            '  "explanation": "Spiegazione della tua scelta"\n'
            "}}\n"
            "```\n"
            "Dove 'recommendations' è una lista di 3 ID numerici dei film e 'explanation' è una breve spiegazione.\n\n"
            "Analizza il catalogo e fornisci le tue raccomandazioni ottimizzate per "
            f"{metric_name.upper()}."
        )
    )

# ----------------------------
# 7. Setup LLM con retry per rate limits
# ----------------------------
llm = ChatOpenAI(model=LLM_MODEL_ID, **COMMON_LLM_PARAMS)

@retry(
    reraise=True,
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError)
)
async def llm_arun_with_retry(prompt_str: str) -> str:
    response = await llm.ainvoke(prompt_str)
    return response

# ----------------------------
# 8. Agent Tools
# ----------------------------
def build_metric_agents():
    """Costruisce un agent per ogni metrica di raccomandazione"""
    metric_tools = []
    for metric_name, metric_desc in PROMPT_VARIANTS.items():
        prompt_template = create_metric_prompt(metric_name, metric_desc)
        
        async def run_metric_agent(catalog, _metric=metric_name, _prompt=prompt_template):
            """Genera raccomandazioni ottimizzate per una specifica metrica"""
            try:
                prompt_str = _prompt.format(catalog=catalog)
                response = await llm_arun_with_retry(prompt_str)
                content = response.content if hasattr(response, 'content') else str(response)
                print(f"Raw response from {_metric} agent: {content}")
                
                # Estrai la parte JSON dalla risposta
                try:
                    import re
                    # Trova qualsiasi blocco JSON con le chiavi che ci interessano
                    json_match = re.search(r'({[\s\S]*?"recommendations"[\s\S]*?"explanation"[\s\S]*?})', content)
                    if json_match:
                        json_str = json_match.group(1)
                        # Sostituisci newline e spazi per normalizzare il JSON
                        json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                        # Rimuovi eventuali caratteri non validi per JSON
                        json_str = re.sub(r'```json|```', '', json_str)
                        json_data = json.loads(json_str)
                        return {
                            "metric": _metric,
                            "recommendations": json_data["recommendations"],
                            "explanation": json_data["explanation"]
                        }
                    
                    # Fallback: cerca solo la lista di raccomandazioni
                    rec_match = re.search(r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]', content)
                    if rec_match:
                        recommendations = json.loads(rec_match.group())
                        return {
                            "metric": _metric,
                            "recommendations": recommendations,
                            "explanation": f"Lista estratta dal testo della risposta: {recommendations}"
                        }
                    
                    # Ultimo tentativo: estrai numeri dal testo
                    numbers = re.findall(r'\b\d+\b', content)
                    if numbers:
                        unique_nums = list(dict.fromkeys([int(n) for n in numbers]))[:3]
                        if len(unique_nums) >= 3:
                            return {
                                "metric": _metric,
                                "recommendations": unique_nums[:3],
                                "explanation": f"Numeri estratti dal testo della risposta: {unique_nums[:3]}"
                            }
                    
                    raise ValueError("Non posso estrarre raccomandazioni dalla risposta")
                    
                except Exception as e:
                    print(f"Error parsing JSON for {_metric}: {str(e)}")
                    return {
                        "metric": _metric,
                        "recommendations": [1, 2, 3],  # Default fallback
                        "explanation": f"Errore nel parsing: {str(e)}. Usando valori predefiniti."
                    }
            except Exception as e:
                print(f"Error running {_metric} agent: {e}")
                return {
                    "metric": _metric,
                    "recommendations": [1, 2, 3],  # Default fallback
                    "explanation": f"Error: {str(e)}"
                }
                
        # Creo una closure per ogni metrica
        metric_fn = (lambda _m=metric_name, _p=prompt_template: 
                    lambda catalog: run_metric_agent(catalog, _m, _p))()
                
        metric_tools.append(
            Tool(
                name=f"recommender_{metric_name}",
                func=metric_fn,
                description=f"Genera raccomandazioni ottimizzate per {metric_name}",
                coroutine=metric_fn
            )
        )
    return metric_tools

# ----------------------------
# 9. Evaluator Tool
# ----------------------------
def build_evaluator_tool():
    """Costruisce un tool per valutare e combinare le raccomandazioni dei vari agenti"""
    eval_prompt = PromptTemplate(
        input_variables=["all_recommendations", "catalog"],
        template=(
            "# Compito: Valutazione e combinazione di raccomandazioni multi-metrica\n\n"
            "Sei un sistema avanzato che deve combinare raccomandazioni di film generate da diversi sistemi specializzati.\n"
            "Ogni sistema si è concentrato su una metrica diversa: precision@k e coverage.\n\n"
            "# Raccomandazioni ricevute:\n"
            "{all_recommendations}\n\n"
            "# Catalogo film:\n"
            "{catalog}\n\n"
            "# Istruzioni:\n"
            "1. Analizza attentamente le raccomandazioni di ciascun sistema\n"
            "2. Considera i punti di forza di ciascuna metrica\n"
            "3. Crea una lista OTTIMALE di 3 film che bilanci le diverse metriche\n"
            "4. Spiega in dettaglio la tua logica e i trade-off considerati\n\n"
            "# Formato di output richiesto:\n"
            "È ESSENZIALE che tu risponda utilizzando ESATTAMENTE questo formato JSON (segui alla lettera lo schema):\n\n"
            "```json\n"
            "{{\n"
            '  "final_recommendations": [145, 270, 381],\n'
            '  "justification": "Spiegazione dettagliata del perché hai scelto questi film",\n'
            '  "trade_offs": "Descrizione dei trade-off considerati tra le diverse metriche"\n'
            "}}\n"
            "```\n\n"
            "IMPORTANTE: Assicurati che il JSON sia valido e contenga TUTTE le chiavi richieste.\n"
            "Le tue final_recommendations DEVONO essere una lista di ESATTAMENTE 3 ID numerici di film.\n"
            "NON aggiungere altre informazioni oltre al JSON richiesto."
        )
    )
    
    async def evaluate_recommendations(all_recommendations_str, catalog_str):
        try:
            prompt_str = eval_prompt.format(
                all_recommendations=all_recommendations_str,
                catalog=catalog_str
            )
            response = await llm_arun_with_retry(prompt_str)
            content = response.content if hasattr(response, 'content') else str(response)
            print(f"Raw evaluator response: {content}")
            
            # Estrai la parte JSON dalla risposta
            try:
                import re
                # Metodo 1: Cerca un oggetto JSON completo con tutte le chiavi richieste
                json_match = re.search(r'({[\s\S]*?"final_recommendations"[\s\S]*?"justification"[\s\S]*?"trade_offs"[\s\S]*?})', content)
                if json_match:
                    json_str = json_match.group(1)
                    # Sostituisci newline e spazi per normalizzare il JSON
                    json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                    # Rimuovi eventuali caratteri non validi per JSON
                    json_str = re.sub(r'```json|```', '', json_str)
                    try:
                        json_data = json.loads(json_str)
                        return {
                            "final_recommendations": json_data.get("final_recommendations", []),
                            "justification": json_data.get("justification", ""),
                            "trade_offs": json_data.get("trade_offs", "")
                        }
                    except json.JSONDecodeError:
                        print(f"JSON malformato: {json_str}")
                
                # Metodo 2: Cerca ciascun campo separatamente
                recommendations = []
                justification = "Non disponibile"
                trade_offs = "Non disponibile"
                
                # Cerca le raccomandazioni
                recs_match = re.search(r'"final_recommendations"\s*:\s*(\[[\s\d,]*\])', content)
                if recs_match:
                    try:
                        recommendations = json.loads(recs_match.group(1))
                    except json.JSONDecodeError:
                        # Fallback: cerca semplicemente una lista in formato JSON
                        rec_match = re.search(r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]', content)
                        if rec_match:
                            try:
                                recommendations = json.loads(rec_match.group())
                            except:
                                pass
                
                # Cerca la giustificazione
                just_match = re.search(r'"justification"\s*:\s*"([^"]*)"', content)
                if just_match:
                    justification = just_match.group(1)
                
                # Cerca i trade-offs
                trade_match = re.search(r'"trade_offs"\s*:\s*"([^"]*)"', content)
                if trade_match:
                    trade_offs = trade_match.group(1)
                
                # Se abbiamo estratto almeno le raccomandazioni, restituisci i risultati
                if recommendations:
                    return {
                        "final_recommendations": recommendations,
                        "justification": justification,
                        "trade_offs": trade_offs
                    }
                
                # Metodo 3: ultimo tentativo con estrazione più libera
                numbers = re.findall(r'\b\d+\b', content)
                if numbers:
                    unique_nums = list(dict.fromkeys([int(n) for n in numbers]))[:3]
                    if len(unique_nums) >= 3:
                        # Cerca di estrarre un testo che assomigli a una giustificazione
                        para_match = re.search(r'\n\n([^.]*?ho scelto[^.]*\.)', content, re.IGNORECASE)
                        justification = para_match.group(1) if para_match else "Estratto dai numeri nel testo"
                        
                        # Cerca di estrarre i trade-offs
                        trade_match = re.search(r'\n\n([^.]*?trade-off[^.]*\.)', content, re.IGNORECASE)
                        trade_offs = trade_match.group(1) if trade_match else "Non disponibile"
                        
                        return {
                            "final_recommendations": unique_nums[:3],
                            "justification": justification,
                            "trade_offs": trade_offs
                        }
                
                print("Non è stato possibile estrarre raccomandazioni dalla risposta")
                return {
                    "final_recommendations": [16, 10, 6],  # Basato sui risultati precedenti
                    "justification": "Non è stato possibile estrarre una giustificazione dalla risposta LLM",
                    "trade_offs": "Non disponibile"
                }
                    
            except Exception as e:
                print(f"Error parsing evaluator JSON: {str(e)}")
                return {
                    "final_recommendations": [16, 10, 6],  # Valori ragionevoli di default basati sui risultati precedenti
                    "justification": f"Errore nel parsing: {str(e)}. Usando valori predefiniti.",
                    "trade_offs": "Non disponibile a causa di errori di parsing"
                }
        except Exception as e:
            print(f"Error in evaluator: {e}")
            return {
                "final_recommendations": [16, 10, 6],  # Valori ragionevoli di default
                "justification": f"Error: {str(e)}",
                "trade_offs": "N/A"
            }
    
    return Tool(
        name="evaluate_recommendations",
        func=evaluate_recommendations,
        description="Valuta e combina raccomandazioni generate dai vari sistemi",
        coroutine=evaluate_recommendations
    )

# ----------------------------
# 10. Runner Agent
# ----------------------------
class RecommenderSystem:
    def __init__(self):
        self.metric_tools = build_metric_agents()
        self.evaluator_tool = build_evaluator_tool()
        self.all_tools = self.metric_tools + [self.evaluator_tool]
        
        self.agent = initialize_agent(
            self.all_tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
    async def run_metric_recommenders(self):
        """Esegue i recommender per ciascuna metrica in parallelo"""
        results = {}
        
        # Ottieni il catalogo ottimizzato
        catalog = get_optimized_catalog(limit=30)
        
        for tool in self.metric_tools:
            metric_name = tool.name.replace("recommender_", "")
            print(f"\n--- Running {metric_name} recommender ---")
            
            try:
                response = await tool.coroutine(catalog)
                results[metric_name] = response
                print(f"Recommendations for {metric_name}: {response['recommendations']}")
                print(f"Explanation: {response['explanation']}")
            except Exception as e:
                print(f"Error running {metric_name} recommender: {e}")
                results[metric_name] = {
                    "metric": metric_name,
                    "recommendations": [],
                    "explanation": f"Error: {str(e)}"
                }
                
        return results
    
    async def evaluate_results(self, metric_results):
        """Valuta e combina i risultati delle diverse metriche"""
        print("\n--- Evaluating combined results ---")
        
        try:
            # Ottieni il catalogo ottimizzato
            catalog = get_optimized_catalog(limit=30)
            
            results_str = json.dumps(metric_results, ensure_ascii=False, indent=2)
            evaluation = await self.evaluator_tool.coroutine(results_str, catalog)
            
            print(f"Final recommendations: {evaluation['final_recommendations']}")
            print(f"Justification: {evaluation['justification']}")
            print(f"Trade-offs: {evaluation['trade_offs']}")
            
            # Calcolo delle metriche
            print("\n--- Calculating metrics ---")
            self.calculate_and_display_metrics(metric_results, evaluation)
            
            return evaluation
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {
                "final_recommendations": [],
                "justification": f"Error in evaluation: {str(e)}",
                "trade_offs": "N/A"
            }
    
    def calculate_and_display_metrics(self, metric_results, final_evaluation):
        """Calcola e visualizza le metriche per le raccomandazioni"""
        try:
            # NUOVO: usa le variabili globali
            global filtered_ratings, user_profiles, movies
            
            # Estrai le raccomandazioni
            precision_at_k_recs = metric_results.get('precision_at_k', {}).get('recommendations', [])
            coverage_recs = metric_results.get('coverage', {}).get('recommendations', [])
            final_recs = final_evaluation.get('final_recommendations', [])
            
            # Prepara i dati per il calcolo delle metriche
            all_movie_ids = movies['movie_id'].tolist()
            
            # Simula dati rilevanti per precision@k (top 100 film più popolari come proxy)
            # In un caso reale, questi sarebbero film che l'utente ha già valutato positivamente
            relevant_items = all_movie_ids[:100]
            
            # Calcola precision@k
            precision_pak_value = calculate_precision_at_k(precision_at_k_recs, relevant_items)
            coverage_pak_value = calculate_precision_at_k(coverage_recs, relevant_items)
            final_pak_value = calculate_precision_at_k(final_recs, relevant_items)
            
            # Calcola coverage
            all_recommendations = [precision_at_k_recs, coverage_recs, final_recs]
            
            # Per la coverage, usiamo il numero di generi unici coperti come approssimazione
            all_genres = set()
            genre_coverage = {}
            
            # Funzione per estrarre i generi di un film
            def get_film_genres(movie_id):
                movie = movies[movies['movie_id'] == movie_id]
                if not movie.empty:
                    genres_str = movie.iloc[0]['genres']
                    return set(genres_str.split('|'))
                return set()
            
            # Calcola i generi unici per ogni set di raccomandazioni
            for name, recs in [("precision_at_k", precision_at_k_recs), 
                              ("coverage", coverage_recs), 
                              ("final", final_recs)]:
                recs_genres = set()
                for movie_id in recs:
                    recs_genres.update(get_film_genres(movie_id))
                
                all_genres.update(recs_genres)
                genre_coverage[name] = len(recs_genres) / (len(all_genres) if all_genres else 1)
            
            # Calcola la coverage totale (film unici raccomandati / totale film)
            all_recommended_ids = set()
            for recs in all_recommendations:
                all_recommended_ids.update(recs)
            total_coverage = len(all_recommended_ids) / len(all_movie_ids)
            
            # Visualizza i risultati
            print("\nMetriche calcolate:")
            print(f"Precision@k per precision_at_k: {precision_pak_value:.4f}")
            print(f"Precision@k per coverage: {coverage_pak_value:.4f}")
            print(f"Precision@k per raccomandazioni finali: {final_pak_value:.4f}")
            
            print(f"\nCoverage per genere (generi unici coperti / totale generi):")
            print(f"  precision_at_k: {genre_coverage.get('precision_at_k', 0):.4f}")
            print(f"  coverage: {genre_coverage.get('coverage', 0):.4f}")
            print(f"  final: {genre_coverage.get('final', 0):.4f}")
            
            print(f"\nCoverage totale (film unici raccomandati / totale film): {total_coverage:.4f}")
            
            # Aggiungi le metriche al risultato finale
            metrics = {
                "precision_at_k": {
                    "precision_score": precision_pak_value,
                    "genre_coverage": genre_coverage.get('precision_at_k', 0)
                },
                "coverage": {
                    "precision_score": coverage_pak_value,
                    "genre_coverage": genre_coverage.get('coverage', 0)
                },
                "final_recommendations": {
                    "precision_score": final_pak_value,
                    "genre_coverage": genre_coverage.get('final', 0)
                },
                "total_coverage": total_coverage
            }
            
            # Aggiungi le metriche ai file di risultati
            with open("recommendation_results.json", "r", encoding="utf-8") as f:
                results = json.load(f)
            
            results["metrics"] = metrics
            
            with open("recommendation_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            return metrics
        except Exception as e:
            print(f"Errore nel calcolo delle metriche: {e}")
            return {}
    
    async def generate_recommendations_with_custom_prompt(self, prompt_variants, experiment_name="custom_experiment"):
        """
        Genera raccomandazioni utilizzando varianti di prompt personalizzate
        
        Args:
            prompt_variants: Dizionario con varianti di prompt per diverse metriche
            experiment_name: Nome dell'esperimento
            
        Returns:
            Tuple con (risultati, nome_file)
        """
        # Salva le varianti di prompt originali
        original_variants = PROMPT_VARIANTS.copy()
        
        try:
            # Sostituisci le varianti di prompt con quelle personalizzate
            for metric, prompt in prompt_variants.items():
                if metric in PROMPT_VARIANTS:
                    PROMPT_VARIANTS[metric] = prompt
                    
            # Ricostruisci gli agent con i nuovi prompt
            self.metric_tools = build_metric_agents()
            self.all_tools = self.metric_tools + [self.evaluator_tool]
            
            # Esegui i recommender per ogni metrica
            metric_results = await self.run_metric_recommenders()
            
            # Valuta e combina i risultati
            final_evaluation = await self.evaluate_results(metric_results)
            
            # Crea il risultato con informazioni sull'esperimento
            result = {
                "timestamp": datetime.now().isoformat(),
                "experiment_info": {
                    "name": experiment_name,
                    "prompt_variants": prompt_variants
                },
                "metric_recommendations": metric_results,
                "final_evaluation": final_evaluation
            }
            
            # Salva il risultato
            os.makedirs("experiments", exist_ok=True)
            filename = f"experiments/experiment_{experiment_name}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            return result, filename
            
        finally:
            # Ripristina le varianti di prompt originali
            for metric, prompt in original_variants.items():
                PROMPT_VARIANTS[metric] = prompt
                
            # Ricostruisci gli agent con i prompt originali
            self.metric_tools = build_metric_agents()
            self.all_tools = self.metric_tools + [self.evaluator_tool]
    
    async def generate_recommendations(self):
        """Genera raccomandazioni eseguendo l'intero pipeline"""
        # Esegui i recommender per ogni metrica
        metric_results = await self.run_metric_recommenders()
        
        # Salva i risultati intermedi
        with open("metric_recommendations.json", "w", encoding="utf-8") as f:
            json.dump(metric_results, f, ensure_ascii=False, indent=2)
            
        # Valuta e combina i risultati
        final_evaluation = await self.evaluate_results(metric_results)
        
        # Salva il risultato finale
        result = {
            "timestamp": datetime.now().isoformat(),
            "metric_recommendations": metric_results,
            "final_evaluation": final_evaluation
        }
        
        with open("recommendation_results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        # Calcola e aggiungi le metriche quantitative
        try:
            # NUOVO: Usa le metriche già calcolate da self.calculate_and_display_metrics
            metrics = self.calculate_and_display_metrics(metric_results, final_evaluation)
            
            # Aggiungi le metriche al risultato
            result["metrics"] = metrics
            
        except ImportError:
            print("Modulo metrics_calculator non trovato. Le metriche non saranno calcolate.")
        except Exception as e:
            print(f"Errore durante il calcolo delle metriche: {e}")
            
        return result

# ----------------------------
# 11. Experiment Reporter
# ----------------------------
class ExperimentReporter:
    """Classe per l'analisi e la generazione di report degli esperimenti di raccomandazione"""
    
    def __init__(self, experiments_dir='experiments'):
        """Inizializza il reporter con la directory degli esperimenti"""
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        self.experiments = []
        self._load_experiments()
    
    def _load_experiments(self):
        """Carica tutti gli esperimenti disponibili nella directory"""
        # Cerca tutti i file JSON nella directory degli esperimenti
        json_files = [f for f in os.listdir(self.experiments_dir) if f.endswith('.json') and f.startswith('experiment_')]
        
        for file in json_files:
            try:
                with open(os.path.join(self.experiments_dir, file), 'r', encoding='utf-8') as f:
                    experiment_data = json.load(f)
                    self.experiments.append(experiment_data)
            except Exception as e:
                print(f"Errore nel caricamento dell'esperimento {file}: {e}")
    
    def add_experiment(self, experiment_data, filename=None):
        """Aggiunge un nuovo esperimento alla collezione"""
        self.experiments.append(experiment_data)
        
        if filename:
            # Copia il file nella directory degli esperimenti se ha un altro percorso
            if not os.path.dirname(filename) == self.experiments_dir:
                target_path = os.path.join(self.experiments_dir, os.path.basename(filename))
                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump(experiment_data, f, ensure_ascii=False, indent=2)
    
    def calculate_diversity_metrics(self):
        """Calcola metriche di diversità tra gli esperimenti"""
        if not self.experiments:
            return {"error": "Nessun esperimento disponibile"}
        
        metrics = {
            "total_experiments": len(self.experiments),
            "unique_recommendations": {},
            "recommendation_frequency": {},
            "metric_performance": {}
        }
        
        # Raccogli tutte le raccomandazioni
        all_recommendations = []
        for exp in self.experiments:
            # Prendi le raccomandazioni finali
            if 'final_evaluation' in exp and 'final_recommendations' in exp['final_evaluation']:
                all_recommendations.extend(exp['final_evaluation']['final_recommendations'])
                
                # Conta la frequenza di ogni raccomandazione
                for rec in exp['final_evaluation']['final_recommendations']:
                    if rec in metrics["recommendation_frequency"]:
                        metrics["recommendation_frequency"][rec] += 1
                    else:
                        metrics["recommendation_frequency"][rec] = 1
        
        # Calcola le raccomandazioni uniche
        metrics["unique_recommendations"] = {
            "count": len(set(all_recommendations)),
            "items": list(set(all_recommendations))
        }
        
        # Analizza le performance per metrica
        for exp in self.experiments:
            if 'metric_recommendations' not in exp:
                continue
                
            for metric, data in exp['metric_recommendations'].items():
                if metric not in metrics["metric_performance"]:
                    metrics["metric_performance"][metric] = {
                        "total_recommendations": [],
                        "explanation_themes": {}
                    }
                
                if 'recommendations' in data:
                    metrics["metric_performance"][metric]["total_recommendations"].extend(data['recommendations'])
                
                # Analisi semplificata delle spiegazioni
                if 'explanation' in data:
                    explanation = data['explanation'].lower()
                    # Estrai temi dalle spiegazioni (semplificato)
                    themes = []
                    if "popolare" in explanation or "popolarità" in explanation:
                        themes.append("popolarità")
                    if "qualità" in explanation:
                        themes.append("qualità")
                    if "diversità" in explanation or "diversi" in explanation:
                        themes.append("diversità")
                    if "genere" in explanation or "generi" in explanation:
                        themes.append("generi")
                    
                    # Incrementa i contatori dei temi
                    for theme in themes:
                        if theme in metrics["metric_performance"][metric]["explanation_themes"]:
                            metrics["metric_performance"][metric]["explanation_themes"][theme] += 1
                        else:
                            metrics["metric_performance"][metric]["explanation_themes"][theme] = 1
        
        # Calcola le raccomandazioni uniche per metrica
        for metric, data in metrics["metric_performance"].items():
            data["unique_recommendations"] = {
                "count": len(set(data["total_recommendations"])),
                "items": list(set(data["total_recommendations"]))
            }
        
        return metrics
    
    def generate_html_report(self, output_file="experiment_report.html"):
        """Genera un report HTML dettagliato degli esperimenti"""
        if not self.experiments:
            return "Nessun esperimento disponibile per generare il report"
        
        diversity_metrics = self.calculate_diversity_metrics()
        
        # Inizia a costruire l'HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Report Esperimenti di Raccomandazione</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .experiment {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .metric {{ margin-bottom: 20px; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }}
                .chart {{ margin: 20px 0; width: 100%; height: 300px; }}
            </style>
        </head>
        <body>
            <h1>Report Esperimenti di Raccomandazione</h1>
            <p>Generato il {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            
            <h2>Statistiche di Diversità</h2>
            <table>
                <tr>
                    <th>Totale Esperimenti</th>
                    <td>{diversity_metrics["total_experiments"]}</td>
                </tr>
                <tr>
                    <th>Raccomandazioni Uniche</th>
                    <td>{diversity_metrics["unique_recommendations"]["count"]}</td>
                </tr>
            </table>
            
            <h3>Film più frequentemente raccomandati</h3>
            <table>
                <tr>
                    <th>ID Film</th>
                    <th>Frequenza</th>
                </tr>
        """
        
        # Aggiungi i film più raccomandati
        sorted_recommendations = sorted(diversity_metrics["recommendation_frequency"].items(), 
                                       key=lambda x: x[1], reverse=True)
        for movie_id, frequency in sorted_recommendations[:10]:  # Top 10
            html += f"""
                <tr>
                    <td>{movie_id}</td>
                    <td>{frequency}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Performance per Metrica</h2>
        """
        
        # Aggiungi statistiche per ogni metrica
        for metric, data in diversity_metrics["metric_performance"].items():
            html += f"""
            <div class="metric">
                <h3>Metrica: {metric}</h3>
                <table>
                    <tr>
                        <th>Raccomandazioni Uniche</th>
                        <td>{data["unique_recommendations"]["count"]}</td>
                    </tr>
                </table>
                
                <h4>Temi nelle Spiegazioni</h4>
                <table>
                    <tr>
                        <th>Tema</th>
                        <th>Frequenza</th>
                    </tr>
            """
            
            # Aggiungi i temi delle spiegazioni
            sorted_themes = sorted(data["explanation_themes"].items(), 
                                 key=lambda x: x[1], reverse=True)
            for theme, frequency in sorted_themes:
                html += f"""
                    <tr>
                        <td>{theme}</td>
                        <td>{frequency}</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        # Aggiungi dettagli per ogni esperimento
        html += """
            <h2>Dettagli Esperimenti</h2>
        """
        
        for i, exp in enumerate(self.experiments):
            exp_name = exp.get('experiment_info', {}).get('name', f'Esperimento {i+1}')
            timestamp = exp.get('timestamp', 'N/A')
            
            html += f"""
            <div class="experiment">
                <h3>{exp_name}</h3>
                <p>Data: {timestamp}</p>
                
                <h4>Varianti di Prompt</h4>
                <pre>{json.dumps(exp.get('experiment_info', {}).get('prompt_variants', {}), indent=2, ensure_ascii=False)}</pre>
                
                <h4>Raccomandazioni per Metrica</h4>
                <table>
                    <tr>
                        <th>Metrica</th>
                        <th>Raccomandazioni</th>
                        <th>Spiegazione</th>
                    </tr>
            """
            
            # Aggiungi le raccomandazioni per metrica
            for metric, data in exp.get('metric_recommendations', {}).items():
                html += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{data.get('recommendations', [])}</td>
                        <td>{data.get('explanation', 'N/A')}</td>
                    </tr>
                """
            
            html += """
                </table>
                
                <h4>Valutazione Finale</h4>
                <table>
                    <tr>
                        <th>Raccomandazioni Finali</th>
                        <td>{}</td>
                    </tr>
                    <tr>
                        <th>Giustificazione</th>
                        <td>{}</td>
                    </tr>
                    <tr>
                        <th>Trade-offs</th>
                        <td>{}</td>
                    </tr>
                </table>
            </div>
            """.format(
                exp.get('final_evaluation', {}).get('final_recommendations', []),
                exp.get('final_evaluation', {}).get('justification', 'N/A'),
                exp.get('final_evaluation', {}).get('trade_offs', 'N/A')
            )
        
        html += """
        </body>
        </html>
        """
        
        # Salva il report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return f"Report salvato in {output_file}"
    
    def generate_markdown_report(self, output_file="experiment_report.md"):
        """Genera un report in formato Markdown degli esperimenti"""
        if not self.experiments:
            return "Nessun esperimento disponibile per generare il report"
        
        diversity_metrics = self.calculate_diversity_metrics()
        
        # Inizia a costruire il Markdown
        md = f"""# Report Esperimenti di Raccomandazione
        
Generato il {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## Statistiche di Diversità

- **Totale Esperimenti**: {diversity_metrics["total_experiments"]}
- **Raccomandazioni Uniche**: {diversity_metrics["unique_recommendations"]["count"]}

### Film più frequentemente raccomandati

| ID Film | Frequenza |
|---------|-----------|
"""
        
        # Aggiungi i film più raccomandati
        sorted_recommendations = sorted(diversity_metrics["recommendation_frequency"].items(), 
                                      key=lambda x: x[1], reverse=True)
        for movie_id, frequency in sorted_recommendations[:10]:  # Top 10
            md += f"| {movie_id} | {frequency} |\n"
        
        md += "\n## Performance per Metrica\n"
        
        # Aggiungi statistiche per ogni metrica
        for metric, data in diversity_metrics["metric_performance"].items():
            md += f"""
### Metrica: {metric}

- **Raccomandazioni Uniche**: {data["unique_recommendations"]["count"]}

#### Temi nelle Spiegazioni

| Tema | Frequenza |
|------|-----------|
"""
            
            # Aggiungi i temi delle spiegazioni
            sorted_themes = sorted(data["explanation_themes"].items(), 
                                 key=lambda x: x[1], reverse=True)
            for theme, frequency in sorted_themes:
                md += f"| {theme} | {frequency} |\n"
            
            md += "\n"
        
        # Aggiungi dettagli per ogni esperimento
        md += "\n## Dettagli Esperimenti\n"
        
        for i, exp in enumerate(self.experiments):
            exp_name = exp.get('experiment_info', {}).get('name', f'Esperimento {i+1}')
            timestamp = exp.get('timestamp', 'N/A')
            
            md += f"""
### {exp_name}

Data: {timestamp}

#### Varianti di Prompt

```json
{json.dumps(exp.get('experiment_info', {}).get('prompt_variants', {}), indent=2, ensure_ascii=False)}
```

#### Raccomandazioni per Metrica

| Metrica | Raccomandazioni | Spiegazione |
|---------|-----------------|-------------|
"""
            
            # Aggiungi le raccomandazioni per metrica
            for metric, data in exp.get('metric_recommendations', {}).items():
                recs = str(data.get('recommendations', []))
                explanation = data.get('explanation', 'N/A').replace('\n', ' ')
                md += f"| {metric} | {recs} | {explanation} |\n"
            
            md += f"""
#### Valutazione Finale

- **Raccomandazioni Finali**: {exp.get('final_evaluation', {}).get('final_recommendations', [])}
- **Giustificazione**: {exp.get('final_evaluation', {}).get('justification', 'N/A').replace('\n', ' ')}
- **Trade-offs**: {exp.get('final_evaluation', {}).get('trade_offs', 'N/A').replace('\n', ' ')}

"""
        
        # Salva il report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md)
        
        return f"Report Markdown salvato in {output_file}"
    
    def compare_prompt_variants(self):
        """Analizza come diverse varianti di prompt influenzano i risultati"""
        if not self.experiments:
            return {"error": "Nessun esperimento disponibile"}
        
        # Raggruppa esperimenti per varianti di prompt
        prompt_groups = {}
        for exp in self.experiments:
            if 'experiment_info' not in exp or 'prompt_variants' not in exp['experiment_info']:
                continue
            
            # Crea una chiave basata sulla variante di prompt
            for metric, prompt in exp['experiment_info']['prompt_variants'].items():
                key = f"{metric}_{hash(prompt)}"
                if key not in prompt_groups:
                    prompt_groups[key] = {
                        "metric": metric,
                        "prompt": prompt,
                        "experiments": [],
                        "recommendations": []
                    }
                
                # Aggiungi l'esperimento al gruppo
                prompt_groups[key]["experiments"].append(exp.get('experiment_info', {}).get('name', 'Unknown'))
                
                # Aggiungi le raccomandazioni
                if 'metric_recommendations' in exp and metric in exp['metric_recommendations']:
                    prompt_groups[key]["recommendations"].extend(
                        exp['metric_recommendations'][metric].get('recommendations', [])
                    )
        
        # Analizza i risultati per ciascun gruppo
        comparison_results = []
        for key, group in prompt_groups.items():
            # Calcola le raccomandazioni uniche
            unique_recs = set(group["recommendations"])
            
            comparison_results.append({
                "metric": group["metric"],
                "prompt_hash": key,
                "prompt_excerpt": group["prompt"][:100] + "..." if len(group["prompt"]) > 100 else group["prompt"],
                "experiments_count": len(group["experiments"]),
                "unique_recommendations": {
                    "count": len(unique_recs),
                    "items": list(unique_recs)
                },
                "experiments": group["experiments"]
            })
        
        return comparison_results
    
    def run_comprehensive_analysis(self, output_dir="reports"):
        """Esegue un'analisi completa e genera report in diversi formati"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Genera i report principali
        html_report = self.generate_html_report(os.path.join(output_dir, "experiment_report.html"))
        md_report = self.generate_markdown_report(os.path.join(output_dir, "experiment_report.md"))
        
        # Esegui analisi aggiuntive
        diversity_metrics = self.calculate_diversity_metrics()
        prompt_comparison = self.compare_prompt_variants()
        
        # Salva i risultati delle analisi in formato JSON
        with open(os.path.join(output_dir, "diversity_metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(diversity_metrics, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(output_dir, "prompt_comparison.json"), 'w', encoding='utf-8') as f:
            json.dump(prompt_comparison, f, ensure_ascii=False, indent=2)
        
        # Genera un report di riepilogo
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.experiments),
            "unique_recommendations": diversity_metrics["unique_recommendations"]["count"],
            "metrics_analyzed": list(diversity_metrics["metric_performance"].keys()),
            "prompt_variants": len(prompt_comparison),
            "reports_generated": [
                os.path.join(output_dir, "experiment_report.html"),
                os.path.join(output_dir, "experiment_report.md"),
                os.path.join(output_dir, "diversity_metrics.json"),
                os.path.join(output_dir, "prompt_comparison.json")
            ]
        }
        
        with open(os.path.join(output_dir, "analysis_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        return summary

# ----------------------------
# 12. Main
# ----------------------------
async def main():
    print("\n=== Starting Metric-Specialized Recommender System ===\n")
    
    # Carica i dataset prima di iniziare
    load_datasets()
    
    # Inizia il processo di raccomandazione
    recommender = RecommenderSystem()
    result = await recommender.generate_recommendations()
    
    print("\n=== Recommendation Process Complete ===")
    print(f"Final recommendations: {result['final_evaluation']['final_recommendations']}")
    
    print("\nResults saved to:")
    print("- metric_recommendations.json (intermediate results)")
    print("- recommendation_results.json (complete results)")

    # Se specificato, esegui esperimenti con varianti di prompt
    run_experiments = os.getenv("RUN_EXPERIMENTS", "false").lower() == "true"
    if run_experiments:
        print("\n=== Running Prompt Variant Experiments ===\n")
        
        # Esempio di varianti di prompt per precision@k
        precision_variants = {
            "precision_at_k_serendipity": (
                "Sei un sistema di raccomandazione esperto che ottimizza per PRECISION@K con focus sulla SERENDIPITY. "
                "Data una lista di film, consiglia i 3 film più rilevanti che potrebbero sorprendere positivamente l'utente. "
                "La precision@k misura la frazione di film raccomandati che l'utente valuterebbe positivamente. "
                "La serendipity si riferisce a raccomandazioni inaspettate ma gradite. "
                "Cerca di bilanciare film popolari con scoperte inaspettate di alta qualità. "
                "NON raccomandare più di 3 film."
            ),
            "precision_at_k_recency": (
                "Sei un sistema di raccomandazione esperto che ottimizza per PRECISION@K con focus sulla RECENCY. "
                "Data una lista di film, consiglia i 3 film più rilevanti e recenti. "
                "La precision@k misura la frazione di film raccomandati che l'utente valuterebbe positivamente. "
                "Considera l'anno di uscita come un fattore importante nella tua decisione. "
                "Concentrati sui film più nuovi di alta qualità. "
                "NON raccomandare più di 3 film."
            )
        }
        
        # Esempio di varianti di prompt per coverage
        coverage_variants = {
            "coverage_genre_balance": (
                "Sei un sistema di raccomandazione esperto che ottimizza per COVERAGE con BILANCIAMENTO DEI GENERI. "
                "Data una lista di film, consiglia 3 film che massimizzano la copertura di diversi generi cinematografici. "
                "La coverage misura la proporzione dell'intero catalogo che il sistema è in grado di raccomandare. "
                "Seleziona film di generi completamente diversi tra loro, evitando sovrapposizioni. "
                "L'obiettivo è rappresentare l'ampiezza del catalogo con sole 3 raccomandazioni. "
                "NON raccomandare più di 3 film."
            ),
            "coverage_temporal": (
                "Sei un sistema di raccomandazione esperto che ottimizza per COVERAGE TEMPORALE. "
                "Data una lista di film, consiglia 3 film che massimizzano la copertura di diverse epoche. "
                "La coverage temporale misura la capacità di raccomandare film di periodi storici diversi. "
                "Seleziona film di decenni diversi, possibilmente distanti tra loro. "
                "L'obiettivo è rappresentare l'ampiezza storica del catalogo con sole 3 raccomandazioni. "
                "NON raccomandare più di 3 film."
            )
        }
        
        # Esegui esperimenti con varianti di precision@k
        for name, prompt in precision_variants.items():
            print(f"\nRunning experiment with prompt variant: {name}")
            variant_dict = {"precision_at_k": prompt}
            result, filename = await recommender.generate_recommendations_with_custom_prompt(
                variant_dict, 
                experiment_name=name
            )
            print(f"Experiment saved to: {filename}")
        
        # Esegui esperimenti con varianti di coverage
        for name, prompt in coverage_variants.items():
            print(f"\nRunning experiment with prompt variant: {name}")
            variant_dict = {"coverage": prompt}
            result, filename = await recommender.generate_recommendations_with_custom_prompt(
                variant_dict, 
                experiment_name=name
            )
            print(f"Experiment saved to: {filename}")
        
        # Esegui un esperimento con entrambe le varianti modificate
        print("\nRunning experiment with multiple prompt variants")
        combined_variants = {
            "precision_at_k": precision_variants["precision_at_k_serendipity"],
            "coverage": coverage_variants["coverage_temporal"]
        }
        result, filename = await recommender.generate_recommendations_with_custom_prompt(
            combined_variants, 
            experiment_name="combined_serendipity_temporal"
        )
        print(f"Combined experiment saved to: {filename}")
        
        # Genera report degli esperimenti
        print("\n=== Generating Experiment Reports ===\n")
        reporter = ExperimentReporter(experiments_dir="experiments")
        
        # Esegui l'analisi completa
        summary = reporter.run_comprehensive_analysis(output_dir="reports")
        
        print("\nAnalysis complete. Reports generated:")
        for report in summary["reports_generated"]:
            print(f"- {report}")
            
        print(f"\nTotal experiments analyzed: {summary['total_experiments']}")
        print(f"Unique recommendations found: {summary['unique_recommendations']}")
        print(f"Metrics analyzed: {', '.join(summary['metrics_analyzed'])}")

if __name__ == "__main__":
    asyncio.run(main())