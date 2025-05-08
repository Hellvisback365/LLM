"""
Gestore dei prompt per il sistema di raccomandazione.
Contiene definizioni dei prompt, funzioni di formattazione e creazione template.
"""

from langchain.prompts import PromptTemplate

# Costante per il numero di raccomandazioni
NUM_RECOMMENDATIONS = 50

# Prompt per diverse metriche
PROMPT_VARIANTS = {
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
        f"1. From the # Movie catalog provided by the user in their message, you MUST select and recommend a list containing EXACTLY {NUM_RECOMMENDATIONS} movie IDs. No more, no less than {NUM_RECOMMENDATIONS}.\\n"
        f"2. The list of {NUM_RECOMMENDATIONS} recommendations MUST be ordered. The first movie ID should be the one you recommend the most (highest probability of positive rating), and the last one the least recommended, based on the user's profile and the provided catalog.\\n"
        f"3. Generating a list with a number of movie IDs different from EXACTLY {NUM_RECOMMENDATIONS} will cause a system error and is strictly forbidden.\\n"
        f"4. Your response MUST include an 'explanation' field (string) detailing the main reasons for your top selections in relation to the user\\\'s profile and the provided movie catalog.\\n"
        "</output_requirements>\\n"
        "<|eot_id|>"
    ),
    "coverage": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n"
        f"You are an expert recommendation system that optimizes for COVERAGE. "
        f"Given a list of movies in the # Movie catalog from the user message, recommend an ORDERED list of EXACTLY {NUM_RECOMMENDATIONS} movies that maximize coverage of different film genres, "
        f"BUT that are still relevant to the specific preferences of the user whose profile you are analyzing. "
        f"Coverage measures the proportion of the entire catalog that the system is able to recommend. "
        f"The goal is to better explore the available movie space and reduce the risk of filter bubbles. "
        f"Make sure your recommendations cover different genres, but are aligned with the user\'s tastes. "
        f"Order the list by putting first the movies that represent a good compromise between genre diversity and user preferences, "
        f"and last those that prioritize pure diversity more at the expense of immediate relevance. "
        f"IMPORTANT: Make specific reference to movies the user has enjoyed to discover related but different genres. "
        f"Each user should receive personalized recommendations based on their unique profile. \\n"
        "<output_requirements>\\n"
        f"1. From the # Movie catalog provided by the user in their message, you MUST select and recommend an ORDERED list of EXACTLY {NUM_RECOMMENDATIONS} movie IDs. This list must maximize genre coverage while remaining relevant to the user's preferences. No more, no less than {NUM_RECOMMENDATIONS} items.\\n"
        f"2. The list of {NUM_RECOMMENDATIONS} recommendations MUST be ordered as described above (compromise between diversity and relevance first, pure diversity last).\\n"
        f"3. It is CRITICAL and MANDATORY that your list contains EXACTLY {NUM_RECOMMENDATIONS} movie IDs. Deviating from this exact number (e.g., providing {NUM_RECOMMENDATIONS-1} or {NUM_RECOMMENDATIONS+1}) will lead to a system failure and is unacceptable.\\n"
        f"4. Your response MUST include an 'explanation' field (string) detailing how your selections achieve genre coverage based on the user's profile and the provided movie catalog.\\n"
        "</output_requirements>\\n"
        "<|eot_id|>"
    )
}

def create_metric_prompt(metric_name: str, metric_description: str) -> PromptTemplate:
    """Crea un PromptTemplate Llama 3.3 formattato per una specifica metrica.
    
    Args:
        metric_name: Il nome della metrica (usato per scopi informativi interni, non nel prompt finale all\'LLM).
        metric_description: Il system prompt Llama 3.3 completo, già formattato con i token
                           <|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>.
    """
    # Il metric_description è il system prompt Llama 3.3 completo (da PROMPT_VARIANTS)
    
    # Contenuto per il messaggio dell'utente
    user_message_content = (
        # Il task specifico è già nel system_prompt (metric_description).
        "# User profile:\\n"
        "{user_profile}\\n\\n"
        "# Movie catalog (use this as the source for your recommendations):\\n"
        "{catalog}\\n\\n"
        "# Required Output Structure (MUST be followed):\\n"
        "<output_format_instructions>\\n"
        f"- The 'recommendations' field MUST be a list of EXACTLY {NUM_RECOMMENDATIONS} integer movie IDs. This count ({NUM_RECOMMENDATIONS}) is absolute, critical, and non-negotiable.\\n"
        f"- The 'recommendations' list MUST be ordered according to the specified metric strategy outlined in the system message.\\n"
        f"- An 'explanation' field (string) detailing the rationale for the {NUM_RECOMMENDATIONS} recommendations MUST be provided.\\n"
        f"- Adherence to providing EXACTLY {NUM_RECOMMENDATIONS} movie IDs is paramount for system functionality. Any deviation will result in failure.\\n"
        "</output_format_instructions>"
    )
    
    # Assembla il template completo per Llama 3.3
    full_prompt_template_str = (
        f"{metric_description}\\n"  # System message (già formattato Llama3)
        "<|start_header_id|>user<|end_header_id|>\\n"
        f"{user_message_content}\\n" # Aggiunto newline alla fine di user_message_content
        "<|eot_id|>\\n"
        "<|start_header_id|>assistant<|end_header_id|>\\n" # Pronto per la risposta dell'LLM
    )
    
    return PromptTemplate(
        input_variables=["catalog", "user_profile"], # Variabili per user_message_content
        template=full_prompt_template_str
    )

def create_evaluation_prompt() -> PromptTemplate:
    """Crea il prompt per la valutazione finale delle raccomandazioni."""
    # Prompt per il sistema
    eval_system_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n"
        "You are an expert recommendation aggregator. Your task is to analyze recommendations from different metrics "
        f"and create an OPTIMAL final recommendation list of EXACTLY {NUM_RECOMMENDATIONS} movies. "
        "Balance the precision@k and coverage metrics, giving more weight to precision for top recommendations while "
        "ensuring adequate coverage of movie genres."
        "\\n\\n"
        "RULES:\\n"
        "1. Balance both metrics, but precision@k should be the primary consideration for top recommendations.\\n"
        "2. Consider the ordering of movies within each metric's list - higher ranked items are more important.\\n"
        "3. Provide a justification that explains your aggregation logic and the trade-offs between metrics.\\n"
        "4. Ensure your final recommendations form a cohesive, personalized set for the user.\\n"
        f"5. STRICT REQUIREMENT: Return EXACTLY {NUM_RECOMMENDATIONS} movie IDs, no more and no less. Your output MUST BE a valid JSON object matching the Pydantic schema.\\n"
        "\\n"
        "<|eot_id|>"
    )
    
    # Template per il messaggio utente  
    user_template_str = (
        "<|start_header_id|>user<|end_header_id|>\\n"
        "# Recommendations per Metric:\\n"
        "{all_recommendations}\\n\\n"
        "# Movie catalog for reference:\\n"
        "{catalog}\\n\\n"
        "# Required Output Format:\\n"
        f"You MUST provide EXACTLY {NUM_RECOMMENDATIONS} movie IDs in your final_recommendations list. "
        "This is a strict requirement - more or fewer IDs will cause a system error. "
        "Include detailed justification and trade-off analysis. Your entire output must be a single, valid JSON object."
        "{feedback_block}" # Placeholder per il feedback
        "\\n<|eot_id|>\\n"
        "<|start_header_id|>assistant<|end_header_id|>\\n"
    )
    
    # Crea il template completo
    return PromptTemplate(
        input_variables=["all_recommendations", "catalog", "feedback_block"],
        template=f"{eval_system_prompt}\\n{user_template_str}"
    ) 