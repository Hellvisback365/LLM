"""
Gestore dei prompt per il sistema di raccomandazione.
Contiene definizioni dei prompt, funzioni di formattazione e creazione template.
"""

from langchain.prompts import PromptTemplate

# Costante per il numero di raccomandazioni
NUM_RECOMMENDATIONS = 50

# Prompt per diverse metriche
# Modificato per usare .format() per NUM_RECOMMENDATIONS successivamente
PROMPT_VARIANTS_TEMPLATES = {
    "precision_at_k": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n"
        "You are a personal movie recommendation consultant optimizing for PRECISION@K for a specific user. "
        "Your goal is to recommend movies that the user will rate 4 or 5 out of 5. "
        "Carefully analyze the user\\\'s profile and focus on the following elements:\\n"
        "1. Genres that the user has consistently rated highly.\\n"
        "2. Identify key actors, directors, themes, and time periods from the user\\\'s appreciated movies. Prioritize movies in the catalog that share these specific attributes.\\n"
        "3. Actively analyze disliked movies to identify genres, themes, or attributes to avoid.\\n\\n"
        "Precision@k measures how many of the recommended movies will actually be rated positively. "
        "When analyzing the catalog, pay particular attention to:\\n"
        "- Genre matching with positively rated movies.\\n"
        "- Thematic and stylistic similarity to favorite movies.\\n"
        "- Avoid movies similar to those the user did not appreciate.\\n\\n"
        "DO NOT recommend movies based on general popularity or trends, unless these "
        "characteristics align with this specific user\\\'s unique preferences. \\n"
        "<output_requirements>\\n"
        f"1. From the # Movie catalog provided by the user in their message, you MUST select and recommend a list containing EXACTLY {{NUM_RECOMMENDATIONS}} movie IDs. No more, no less than {{NUM_RECOMMENDATIONS}}.\\n"
        f"2. The list of {{NUM_RECOMMENDATIONS}} recommendations MUST be ordered. The first movie ID should be the one you recommend the most (highest probability of positive rating), and the last one the least recommended, based on the user's profile and the provided catalog.\\n"
        f"3. Generating a list with a number of movie IDs different from EXACTLY {{NUM_RECOMMENDATIONS}} will cause a system error and is strictly forbidden. IT IS ABSOLUTELY CRITICAL that the count is precise. Double-check your output count before responding.\\n"
        "4. Your response MUST include an 'explanation' field (string) detailing the main reasons for your top selections in relation to the user\\\'s profile and the provided movie catalog.\\n"
        "</output_requirements>\\n"
        "<|eot_id|>"
    ),
    "coverage": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n"
        "You are an expert recommendation system that optimizes for COVERAGE. "
        "Given a list of movies in the # Movie catalog from the user message, recommend an ORDERED list of EXACTLY {NUM_RECOMMENDATIONS} movies that maximize coverage of different film genres, "
        "BUT that are still relevant to the specific preferences of the user whose profile you are analyzing. "
        "Coverage measures the proportion of the entire catalog that the system is able to recommend. "
        "The goal is to better explore the available movie space and reduce the risk of filter bubbles. "
        "Make sure your recommendations cover different genres, but are aligned with the user\\'s tastes. "
        "Order the list by putting first the movies that represent a good compromise between genre diversity and user preferences, "
        "and last those that prioritize pure diversity more at the expense of immediate relevance. "
        "IMPORTANT: Make specific reference to movies the user has enjoyed to discover related but different genres. "
        "Each user should receive personalized recommendations based on their unique profile. \\n"
        "<output_requirements>\\n"
        f"1. From the # Movie catalog provided by the user in their message, you MUST select and recommend an ORDERED list of EXACTLY {{NUM_RECOMMENDATIONS}} movie IDs. This list must maximize genre coverage while remaining relevant to the user's preferences. No more, no less than {{NUM_RECOMMENDATIONS}} items.\\n"
        f"2. The list of {{NUM_RECOMMENDATIONS}} recommendations MUST be ordered as described above (compromise between diversity and relevance first, pure diversity last).\\n"
        f"3. It is CRITICAL AND MANDATORY that your list contains EXACTLY {{NUM_RECOMMENDATIONS}} movie IDs. Deviating from this exact number (e.g., providing {{NUM_RECOMMENDATIONS_MINUS_1}} or {{NUM_RECOMMENDATIONS_PLUS_1}}) will lead to a system failure and is unacceptable. THIS IS A NON-NEGOTIABLE REQUIREMENT. Confirm your count is {{NUM_RECOMMENDATIONS}} before responding.\\n" 
        "4. Your response MUST include an 'explanation' field (string) detailing how your selections achieve genre coverage based on the user's profile and the provided movie catalog.\\n"
        "</output_requirements>\\n"
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
    user_message_content_template = """# User profile:
{{user_profile}}

# Movie catalog (use this as the source for your recommendations):
{{catalog}}

# Required Output Structure (MUST be followed):
<output_format_instructions>
- The 'recommendations' field MUST be a list of EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} integer movie IDs. This count ({NUM_RECOMMENDATIONS_PLACEHOLDER}) is absolute, critical, and non-negotiable. Any deviation will result in failure. YOU MUST DOUBLE-CHECK THIS COUNT.
- The 'recommendations' list MUST be ordered according to the specified metric strategy outlined in the system message.
- An 'explanation' field (string) detailing the rationale for the {NUM_RECOMMENDATIONS_PLACEHOLDER} recommendations MUST be provided.
- Adherence to providing EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} movie IDs is paramount for system functionality. Any deviation will result in failure.
- Your entire response MUST be a single JSON object enclosed in triple backticks.
  The JSON object MUST have a key "recommendations" which is a list of EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} integer movie IDs, and a key "explanation" which is a string.
  Example of the required JSON structure (ensure EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} items in 'recommendations'):
  ```json
  {{{{  // Quadruple braces
    "recommendations": [
      movie_id_1, // first recommended movie ID
      movie_id_2,
      // ... (additional movie IDs up to {NUM_RECOMMENDATIONS_PLACEHOLDER}) ...
      movie_id_{NUM_RECOMMENDATIONS_PLACEHOLDER} // last of the {NUM_RECOMMENDATIONS_PLACEHOLDER} movie IDs
    ],
    "explanation": "Your detailed explanation for why these specific {NUM_RECOMMENDATIONS_PLACEHOLDER} movies were chosen and ordered."
  }}}}  // Quadruple braces
  ```
- Do not include any other text, preambles, or explanations outside the main triple backtick block.
</output_format_instructions>"""
    
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
        "You are an expert recommendation aggregator. Your task is to analyze recommendations from different metrics "
        "and create an OPTIMAL final recommendation list of EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} movies. "
        "Balance the precision@k and coverage metrics, giving more weight to precision for top recommendations while "
        "ensuring adequate coverage of movie genres."
        "\\n\\n"
        "RULES:\\n"
        "1. Balance both metrics, but precision@k should be the primary consideration for top recommendations.\\n"
        "2. Consider the ordering of movies within each metric's list - higher ranked items are more important.\\n"
        "3. Provide a justification that explains your aggregation logic and the trade-offs between metrics.\\n"
        "4. Ensure your final recommendations form a cohesive, personalized set for the user.\\n"
        "5. STRICT REQUIREMENT: Return EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} movie IDs, no more and no less. Your output MUST BE a valid JSON object matching the Pydantic schema.\\n"
        "\\n"
        "<|eot_id|>"
    )
    # Formatta con il valore effettivo di NUM_RECOMMENDATIONS
    eval_system_prompt = eval_system_prompt_template.format(NUM_RECOMMENDATIONS_PLACEHOLDER=NUM_RECOMMENDATIONS)
    
    # user_template_str_template usa {NUM_RECOMMENDATIONS_PLACEHOLDER}
    user_template_str_template = """<|start_header_id|>user<|end_header_id|>
# Recommendations per Metric:
{{all_recommendations}}

# Movie catalog for reference:
{{catalog}}

# Required Output Format:
You MUST provide EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} movie IDs in your final_recommendations list. This is a strict requirement - more or fewer IDs will cause a system error. Include detailed justification and trade-off analysis. Your entire output must be a single, valid JSON object.{{feedback_block}}
Your entire response MUST be a single JSON object enclosed in triple backticks.
  The JSON object MUST have keys "final_recommendations" (a list of EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} integer movie IDs), "justification" (string), and "trade_offs" (string).
  Example of the required JSON structure (ensure EXACTLY {NUM_RECOMMENDATIONS_PLACEHOLDER} items in 'final_recommendations'):
  ```json
  {{{{  // Quadruple braces
    "final_recommendations": [
      movie_id_1, // first recommended movie ID
      movie_id_2,
      // ... (additional movie IDs up to {NUM_RECOMMENDATIONS_PLACEHOLDER}) ...
      movie_id_{NUM_RECOMMENDATIONS_PLACEHOLDER} // last of the {NUM_RECOMMENDATIONS_PLACEHOLDER} movie IDs
    ],
    "justification": "Detailed justification for the final selection and ordering.",
    "trade_offs": "Description of trade-offs considered between metrics."
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