"""
Servizio di interfaccia con l'API di OpenRouter per il modello LLM.
Gestisce la comunicazione con i modelli LLM per generare raccomandazioni.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Setup ambiente e parametri
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class LLMService:
    """
    Classe per gestire l'interazione con i modelli LLM tramite OpenRouter API.
    """
    
    def __init__(self, model_name: str = "openai/gpt-4o-mini", temperature: float = 0.7):
        """
        Inizializza il servizio LLM.
        
        Args:
            model_name: Nome del modello LLM da utilizzare
            temperature: Temperatura per la generazione (0-1)
        """
        if not OPENROUTER_API_KEY:
            raise ValueError("Missing OPENROUTER_API_KEY in environment")
            
        self.model_name = model_name
        self.common_params = {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": OPENROUTER_API_KEY,
            "temperature": temperature,
            "max_tokens": 512,
        }
        
    def _create_llm(self) -> ChatOpenAI:
        """Crea un'istanza del modello LLM con i parametri configurati"""
        return ChatOpenAI(model=self.model_name, **self.common_params)
        
    def make_prompt_with_parser(self, metric_key: str, metric_text: str, example: str = None) -> Tuple[PromptTemplate, StructuredOutputParser]:
        """
        Costruisce un prompt per una metrica specifica con relativo parser.
        
        Args:
            metric_key: Chiave identificativa della metrica
            metric_text: Testo descrittivo della metrica
            example: Esempio di output atteso (opzionale)
            
        Returns:
            Tupla con (prompt_template, output_parser)
        """
        schema = ResponseSchema(
            name="recommendations",
            description="Lista JSON di ID numerici delle raccomandazioni, in ordine di priorità."
        )
        parser = StructuredOutputParser.from_response_schemas([schema])
        fmt_instr = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
        
        # Aggiungi esempio se fornito
        if example:
            fmt_instr = f"Esempio corretto: {example}\n" + fmt_instr

        template_str = (
            f"Metric: {metric_key}\n"
            f"Descrizione: {metric_text}\n"
            f"Catalogo: {{catalog}}\n"
            f"Profilo Utente: {{user_profile}}\n\n"
            f"ISTRUZIONI IMPORTANTI PER LA POPOLARITÀ DEI FILM:\n"
            f"1. IGNORA completamente l'ID numerico del film nel valutare la sua popolarità.\n"
            f"2. Valuta la popolarità di ciascun film basandoti sulla tua conoscenza generale del cinema e della cultura popolare.\n"
            f"3. Considera fattori come: titolo riconoscibile, franchise famosi, anno di uscita, registi e attori noti.\n"
            f"4. Per film con titoli simili a film famosi del mondo reale, presumi che siano gli stessi film.\n"
            f"5. Quando la metrica richiede film popolari, scegli i film che ritieni più conosciuti e amati dal pubblico generale.\n\n"
            f"{fmt_instr}"
        )
        return PromptTemplate(input_variables=["catalog", "user_profile"], template=template_str), parser
    
    def build_metric_chains(self, metrics_definitions: Dict[str, str]) -> Tuple[Dict, Dict, Dict]:
        """
        Costruisce le catene di prompt per tutte le metriche definite.
        
        Args:
            metrics_definitions: Dizionario con definizioni delle metriche
            
        Returns:
            Tuple con (chains, parsers, raw_prompts)
        """
        llm = self._create_llm()
        chains, parsers, raw_prompts = {}, {}, {}
        
        for metric, description in metrics_definitions.items():
            # Crea un esempio di output in base alla metrica
            example = "[1, 2, 3]"  # Esempio di default
            
            prompt, parser = self.make_prompt_with_parser(metric, description, example)
            chains[metric] = prompt | llm
            parsers[metric] = parser
            raw_prompts[metric] = prompt.template
            
        return chains, parsers, raw_prompts
    
    def build_evaluator_chain(self) -> Tuple[Any, StructuredOutputParser]:
        """
        Costruisce la catena di valutazione per combinare i risultati di diverse metriche.
        
        Returns:
            Tuple con (evaluator_chain, evaluator_parser)
        """
        llm = self._create_llm()
     
        final_recs_schema = ResponseSchema( 
            name="final_recommendations", 
            description="Lista JSON ottimizzata finale di ID numerici, basata sull'analisi di tutte le metriche." 
        )
        justification_schema = ResponseSchema(
            name="justification",
            description="Spiegazione testuale della selezione finale delle raccomandazioni."
        )
        parser = StructuredOutputParser.from_response_schemas([final_recs_schema, justification_schema])
        fmt_instr = parser.get_format_instructions().replace("{","{{").replace("}","}}")

        prompt = PromptTemplate(
            input_variables=["metrics_results_json"], 
            template=(
                "Sei un sistema di raccomandazione che ottimizza per diversi obiettivi.\n"
                "Ti vengono forniti i risultati di raccomandazioni generate per diverse metriche:\n"
                "- accuracy: suggerimenti precisi basati sulle preferenze dell'utente\n"
                "- diversity: suggerimenti che coprono diversi generi e categorie\n"
                "- novelty: suggerimenti di elementi originali e meno conosciuti\n\n"
                "Dati delle metriche:\n"
                "```json\n"
                "{metrics_results_json}\n"
                "```\n\n"
                "ISTRUZIONI:\n"
                "1. Analizza i risultati e le raccomandazioni di tutte le metriche\n"
                "2. Crea una lista finale di raccomandazioni ottimale che bilanci le diverse metriche\n"
                "3. Fornisci una spiegazione della tua scelta finale\n\n"
                "Formato di output richiesto:\n"
                f"{fmt_instr}"
            )
        )

        return prompt | llm, parser
        
    async def generate_recommendations(self, chain: Any, parser: StructuredOutputParser, 
                                catalog: str, user_profile: str) -> Tuple[List[int], str]:
        """
        Genera raccomandazioni utilizzando una catena LLM.
        
        Args:
            chain: Catena LLM da utilizzare
            parser: Parser per l'output
            catalog: Catalogo di elementi in formato stringa
            user_profile: Profilo utente in formato stringa
            
        Returns:
            Tupla con (lista_id_raccomandati, output_grezzo)
        """
        try: 
            result = await chain.ainvoke({
                "catalog": catalog,
                "user_profile": user_profile
            })
        except Exception as e:
            print(f"LLM error: {e}")
            return [], ""
            
        content = getattr(result, "content", result)
        raw = "\n".join(content.splitlines()[:5]) + ("..." if len(content.splitlines()) > 5 else "")
        
        try:
            output = parser.parse(content)
            ids = output.get("recommendations", [])
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
                
        # Cast a int e pulizia
        cleaned = []
        for x in ids:
            try:
                cleaned.append(int(x))
            except:
                pass
                
        return cleaned, raw
        
    async def evaluate_recommendations(self, evaluator_chain: Any, evaluator_parser: StructuredOutputParser, 
                                metrics_results: Dict) -> Dict:
        """
        Valuta i risultati delle diverse metriche per generare raccomandazioni finali.
        
        Args:
            evaluator_chain: Catena LLM per la valutazione
            evaluator_parser: Parser per l'output
            metrics_results: Risultati delle metriche in formato dizionario
            
        Returns:
            Dizionario con raccomandazioni finali e giustificazione
        """
        try:
            metrics_json = json.dumps(metrics_results, ensure_ascii=False)
            result = await evaluator_chain.ainvoke({"metrics_results_json": metrics_json})
            
            content = getattr(result, "content", result)
            evaluation = evaluator_parser.parse(content)
            
            # Garantisci che final_recommendations sia una lista di interi
            recommendations = evaluation.get("final_recommendations", [])
            if isinstance(recommendations, str):
                try:
                    recommendations = json.loads(recommendations)
                except:
                    recommendations = []
                    
            cleaned_recs = []
            for x in recommendations:
                try:
                    cleaned_recs.append(int(x))
                except:
                    pass
                    
            return {
                "final_recommendations": cleaned_recs,
                "justification": evaluation.get("justification", "")
            }
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {
                "final_recommendations": [],
                "justification": f"Error during evaluation: {str(e)}"
            } 