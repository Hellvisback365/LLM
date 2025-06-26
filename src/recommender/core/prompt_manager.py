"""
Gestore dei prompt per il sistema di raccomandazione.
Contiene definizioni dei prompt, funzioni di formattazione e creazione template.
"""

from langchain.prompts import PromptTemplate

# Costante per il numero di raccomandazioni - ridotto per stabilità LLM
NUM_RECOMMENDATIONS = 20

# Prompt per diverse metriche
# Modificato per usare .format() per NUM_RECOMMENDATIONS successivamente
PROMPT_VARIANTS_TEMPLATES = {
    "precision_at_k": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n"
        "You are a movie recommendation system optimizing for PRECISION@K. "
        "Recommend movies the user will rate 4-5/5 based on their profile. "
        "Focus on: 1) Genres user likes, 2) Similar actors/directors, 3) Avoid disliked patterns.\\n"
        f"OUTPUT: JSON with 'recommendations' array of EXACTLY {{NUM_RECOMMENDATIONS}} movie IDs and 'explanation' string.\\n"
        "<|eot_id|>"
    ),
    "coverage": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n"
        "You are a movie recommendation system optimizing for COVERAGE. "
        "Maximize genre diversity - recommend from different genres, time periods, and styles. "
        "Focus on exploration over user satisfaction.\\n"
        f"OUTPUT: JSON with 'recommendations' array of EXACTLY {{NUM_RECOMMENDATIONS}} movie IDs and 'explanation' string.\\n"
        "<|eot_id|>"
    )
}

# Formatta i PROMPT_VARIANTS con NUM_RECOMMENDATIONS
PROMPT_VARIANTS = {
    metric_name: template.format(
        NUM_RECOMMENDATIONS=NUM_RECOMMENDATIONS,
        NUM_RECOMMENDATIONS_MINUS_1=NUM_RECOMMENDATIONS - 1, # Aggiunto per il prompt coverage
        NUM_RECOMMENDATIONS_PLUS_1=NUM_RECOMMENDATIONS + 1   # Aggiunto per il prompt coverage
    )
    for metric_name, template in PROMPT_VARIANTS_TEMPLATES.items()
}

def create_metric_prompt(metric_name: str, metric_description: str) -> PromptTemplate:
    """Crea un PromptTemplate Llama 3.3 formattato per una specifica metrica.
    
    Args:
        metric_name: Il nome della metrica (usato per scopi informativi interni, non nel prompt finale all'LLM).
        metric_description: Il system prompt Llama 3.3 completo, già formattato con i token
                           <|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>.
                           E NUM_RECOMMENDATIONS già inserito.
    """
    
    # user_message_content_template ora usa {NUM_RECOMMENDATIONS_PLACEHOLDER}
    # che verrà formattato con .format() prima di creare PromptTemplate
    user_message_content_template = """User Profile: {{user_profile}}

Movie Catalog: {{catalog}}

Output (JSON only):
```json
{{{{
  "recommendations": [list of EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} integer movie IDs],
  "explanation": "Brief explanation"
}}}}
```"""
    
    # Formatta user_message_content_template con il valore effettivo di NUM_RECOMMENDATIONS
    user_message_content = user_message_content_template.format(NUM_RECOMMENDATIONS_PLACEHOLDER=NUM_RECOMMENDATIONS)
    
    # Assembla il template completo per Llama 3.3
    full_prompt_template_str = (
        f"{metric_description}\\n" # metric_description ha già NUM_RECOMMENDATIONS inserito
        "<|start_header_id|>user<|end_header_id|>\\n"
        f"{user_message_content}\\n" # user_message_content ha già NUM_RECOMMENDATIONS inserito
        "<|eot_id|>\\n"
        "<|start_header_id|>assistant<|end_header_id|>\\n"
    )
    
    return PromptTemplate(
        input_variables=["catalog", "user_profile"], 
        template=full_prompt_template_str
    )

def create_evaluation_prompt() -> PromptTemplate:
    """Crea il prompt per la valutazione finale delle raccomandazioni."""
    
    # eval_system_prompt_template usa {NUM_RECOMMENDATIONS_PLACEHOLDER}
    eval_system_prompt_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n"
        "You are a recommendation aggregator. Create a final list of EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} movies "
        "by combining precision@k and coverage recommendations from multiple users. "
        "Balance user preferences with genre diversity.\\n"
        "OUTPUT: JSON with 'final_recommendations' array of {NUM_RECOMMENDATIONS_PLACEHOLDER} movie IDs, 'justification' and 'trade_offs' strings.\\n"
        "<|eot_id|>"
    )
    # Formatta con il valore effettivo di NUM_RECOMMENDATIONS
    eval_system_prompt = eval_system_prompt_template.format(NUM_RECOMMENDATIONS_PLACEHOLDER=NUM_RECOMMENDATIONS)
    
    # user_template_str_template usa {NUM_RECOMMENDATIONS_PLACEHOLDER}
    user_template_str_template = """<|start_header_id|>user<|end_header_id|>
Recommendations: {{all_recommendations}}

Catalog: {{catalog}}

Output (JSON only):
```json
{{{{
  "final_recommendations": [EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} movie IDs],
  "justification": "Brief explanation",
  "trade_offs": "Brief trade-off analysis"
}}}}
```
  ```json
  {{{{  // Quadruple braces
    "final_recommendations": [
      movie_id_1, // first recommended movie ID
      movie_id_2,
      // ... (additional movie IDs up to {NUM_RECOMMENDATIONS_PLACEHOLDER}) ...
      movie_id_{NUM_RECOMMENDATIONS_PLACEHOLDER} // last of the {NUM_RECOMMENDATIONS_PLACEHOLDER} movie IDs
    ],
    "justification": "Detailed justification for the final selection and ordering, explaining how you aggregated across users and metrics.",
    "trade_offs": "Description of trade-offs considered between precision_at_k and coverage metrics."
  }}}}  // Quadruple braces
  ```
  Do not include any other text, preambles, or explanations outside the main triple backtick block. THE COUNT OF 'final_recommendations' MUST BE EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER}.

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    # Formatta con il valore effettivo di NUM_RECOMMENDATIONS
    user_template_str = user_template_str_template.format(NUM_RECOMMENDATIONS_PLACEHOLDER=NUM_RECOMMENDATIONS)
    
    # Crea il template completo
    return PromptTemplate(
        input_variables=["all_recommendations", "catalog", "feedback_block"],
        template=f"{eval_system_prompt}\\n{user_template_str}"
    ) 