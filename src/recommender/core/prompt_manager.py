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
        "You are an expert movie recommendation system optimizing for PRECISION@K. "
        "Your goal is to recommend movies the user will DEFINITELY love (rate 4-5/5). "
        "ANALYSIS STRATEGY:\\n"
        "1. GENRE ANALYSIS: Identify user's top 2-3 preferred genres from liked_movies\\n"
        "2. QUALITY FOCUS: Recommend highly-rated movies (avg_rating >= 4.0) in preferred genres\\n"
        "3. PATTERN MATCHING: Find movies similar to user's liked_movies in style/theme\\n"
        "4. AVOID RISKS: Skip experimental or niche movies that might disappoint\\n"
        "5. SAFE CHOICES: Prioritize popular, well-reviewed movies in user's preferred genres\\n"
        f"OUTPUT: JSON with 'recommendations' array of EXACTLY {{NUM_RECOMMENDATIONS}} movie IDs and 'explanation' string.\\n"
        "<|eot_id|>"
    ),
    "coverage": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n"
        "You are an expert movie recommendation system optimizing for COVERAGE. "
        "Your goal is to maximize genre and style diversity while maintaining quality. "
        "DIVERSITY STRATEGY:\\n"
        "1. GENRE SPREAD: Include movies from AT LEAST 8 different genres\\n"
        "2. TIME PERIODS: Mix classics (pre-1990), 90s hits, 2000s favorites, and recent films\\n"
        "3. STYLE VARIETY: Include different movie types (action, drama, comedy, thriller, sci-fi, etc.)\\n"
        "4. INTERNATIONAL: Consider foreign films and different cultural perspectives\\n"
        "5. BALANCE: Still prefer higher-rated movies (avg_rating >= 3.5) for quality\\n"
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
    user_message_content_template = """ANALYSIS TASK:
Analyze the user profile and movie catalog to make {NUM_RECOMMENDATIONS_PLACEHOLDER} high-quality recommendations.

USER PROFILE: {{user_profile}}

AVAILABLE MOVIES: {{catalog}}

STEP-BY-STEP ANALYSIS:
1. Analyze user's liked_movies to identify preferred genres, themes, and patterns
2. Check user's disliked_movies to understand what to avoid
3. Filter catalog movies by user's preferred criteria
4. Select the best {NUM_RECOMMENDATIONS_PLACEHOLDER} movies that match the optimization goal

OUTPUT ONLY THIS JSON:
```json
{{{{
  "recommendations": [list of EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} integer movie IDs from the catalog],
  "explanation": "Brief analysis of selection strategy and patterns found"
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