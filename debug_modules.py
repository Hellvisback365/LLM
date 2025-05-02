import os
import sys
import json
import inspect
import asyncio
from functools import wraps
from datetime import datetime

# Add the project root to the Python path if needed
sys.path.append('.')

# Import the modules we want to trace
from src.recommender.api.llm_service import LLMService
from src.recommender.core.metrics_calculator import calculate_metrics_for_recommendations, add_metrics_to_results
from src.recommender.core.recommender import RecommenderSystem

# Create a log file for tracing
LOG_FILE = "module_debug_trace.log"

def log_message(message):
    """Write a message to the log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    
    # Also print to console
    print(f"[{timestamp}] {message}")

def trace_function(func):
    """Decorator to trace function calls and returns"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        log_message(f"CALL {func.__module__}.{func.__name__}")
        result = func(*args, **kwargs)
        log_message(f"RETURN {func.__module__}.{func.__name__} -> {type(result)}")
        return result
    return wrapper

async def trace_async_function(func):
    """Decorator to trace async function calls and returns"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        log_message(f"ASYNC CALL {func.__module__}.{func.__name__}")
        result = await func(*args, **kwargs)
        log_message(f"ASYNC RETURN {func.__module__}.{func.__name__} -> {type(result)}")
        return result
    return wrapper

# Apply tracing to key LLMService methods
LLMService._create_llm = trace_function(LLMService._create_llm)
LLMService.make_prompt_with_parser = trace_function(LLMService.make_prompt_with_parser)
LLMService.build_metric_chains = trace_function(LLMService.build_metric_chains)
LLMService.build_evaluator_chain = trace_function(LLMService.build_evaluator_chain)

# Wrap async methods
original_generate_recommendations = LLMService.generate_recommendations
@wraps(original_generate_recommendations)
async def traced_generate_recommendations(self, *args, **kwargs):
    log_message(f"ASYNC CALL LLMService.generate_recommendations")
    result = await original_generate_recommendations(self, *args, **kwargs)
    log_message(f"ASYNC RETURN LLMService.generate_recommendations -> {type(result)}")
    return result
LLMService.generate_recommendations = traced_generate_recommendations

original_evaluate_recommendations = LLMService.evaluate_recommendations
@wraps(original_evaluate_recommendations)
async def traced_evaluate_recommendations(self, *args, **kwargs):
    log_message(f"ASYNC CALL LLMService.evaluate_recommendations")
    result = await original_evaluate_recommendations(self, *args, **kwargs)
    log_message(f"ASYNC RETURN LLMService.evaluate_recommendations -> {type(result)}")
    return result
LLMService.evaluate_recommendations = traced_evaluate_recommendations

# Apply tracing to metrics_calculator functions
original_calculate_metrics = calculate_metrics_for_recommendations
@wraps(original_calculate_metrics)
def traced_calculate_metrics(*args, **kwargs):
    log_message(f"CALL calculate_metrics_for_recommendations")
    result = original_calculate_metrics(*args, **kwargs)
    log_message(f"RETURN calculate_metrics_for_recommendations -> {type(result)}")
    return result
sys.modules['src.recommender.core.metrics_calculator'].calculate_metrics_for_recommendations = traced_calculate_metrics

original_add_metrics = add_metrics_to_results
@wraps(original_add_metrics)
def traced_add_metrics(*args, **kwargs):
    log_message(f"CALL add_metrics_to_results")
    result = original_add_metrics(*args, **kwargs)
    log_message(f"RETURN add_metrics_to_results -> {result}")
    return result
sys.modules['src.recommender.core.metrics_calculator'].add_metrics_to_results = traced_add_metrics

# Create a simplified test of the recommender system
async def run_simplified_test():
    """Run a simplified test of the recommender system with tracing"""
    from agent import load_datasets
    
    # Initialize log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== MODULE DEBUG TRACE - {datetime.now().isoformat()} ===\n\n")
    
    log_message("Starting simplified test of recommender system")
    
    try:
        # Load datasets
        log_message("Loading datasets")
        filtered_ratings, user_profiles, movies = load_datasets()
        log_message(f"Loaded data: {len(movies)} movies, {len(user_profiles)} user profiles")
        
        # Create a test catalog
        log_message("Creating test catalog from the first 20 movies")
        test_catalog = movies.head(20).to_dict('records')
        catalog_json = json.dumps(test_catalog, ensure_ascii=False)
        
        # Save the movies catalog for metrics_calculator to use
        log_message("Saving movies catalog for metrics_calculator")
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        with open(os.path.join("data", "processed", "movies_catalog.json"), "w", encoding="utf-8") as f:
            json.dump(test_catalog, f, ensure_ascii=False, indent=2)
        
        # Create test user profile
        log_message("Creating test user profile")
        user_id = 1  # Use the first user
        user_profile = user_profiles.loc[user_id].to_dict()
        user_profile_json = json.dumps({
            "user_id": int(user_id),
            "liked_movies": user_profile.get("liked_movies", []),
            "disliked_movies": user_profile.get("disliked_movies", [])
        }, ensure_ascii=False)
        
        # Create a LLM service instance
        log_message("Creating LLMService instance")
        llm_service = LLMService(model_name="mistralai/mistral-large-2411", temperature=0.7)
        
        # Create metric definitions
        metrics_definitions = {
            "accuracy": "Recommend films the user will like based on their preferences",
            "diversity": "Recommend films from different genres and categories"
        }
        
        # Build metric chains
        log_message("Building metric chains")
        chains, parsers, raw_prompts = llm_service.build_metric_chains(metrics_definitions)
        
        # Run generate_recommendations for one metric
        log_message("Generating recommendations for accuracy metric")
        accuracy_chain = chains["accuracy"]
        accuracy_parser = parsers["accuracy"]
        accuracy_recs, raw_output = await llm_service.generate_recommendations(
            accuracy_chain, accuracy_parser, catalog_json, user_profile_json
        )
        
        log_message(f"Accuracy recommendations: {accuracy_recs}")
        
        # Run evaluator chain
        log_message("Building evaluator chain")
        evaluator_chain, evaluator_parser = llm_service.build_evaluator_chain()
        
        # Create metrics results
        metrics_results = {
            "accuracy": {
                "recommendations": accuracy_recs,
                "explanation": "Based on user preferences"
            },
            "diversity": {
                "recommendations": [4, 5, 6],  # Use mock data for diversity
                "explanation": "Diverse set of genres"
            }
        }
        
        # Run evaluator
        log_message("Evaluating recommendations")
        evaluation = await llm_service.evaluate_recommendations(
            evaluator_chain, evaluator_parser, metrics_results
        )
        
        log_message(f"Evaluation result: {evaluation}")
        
        # Create a results file
        result = {
            "timestamp": datetime.now().isoformat(),
            "metric_recommendations": metrics_results,
            "final_evaluation": evaluation
        }
        
        with open("debug_recommendation_results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        # Also create a mock ratings file for the metrics calculation
        log_message("Creating mock ratings data for metrics calculation")
        ratings_data = []
        for movie_id in range(1, 21):  # First 20 movies
            # Add some mock ratings
            ratings_data.append({"user_id": 1, "movie_id": movie_id, "rating": 4.5 if movie_id in [1, 2, 6, 10, 15, 19] else 3.0})
            
        # Save the mock ratings
        with open("mock_ratings.json", "w", encoding="utf-8") as f:
            json.dump(ratings_data, f, ensure_ascii=False, indent=2)
            
        # Use metrics_calculator
        log_message("Calculating metrics")
        metrics = calculate_metrics_for_recommendations(metrics_results, evaluation)
        
        log_message(f"Metrics calculated: {list(metrics.keys() if isinstance(metrics, dict) else [])}")
        
        # Add metrics to results
        log_message("Adding metrics to results file")
        success = add_metrics_to_results(metrics, output_file="debug_recommendation_results.json")
        
        log_message(f"Metrics added to results: {success}")
        
        # Summary
        log_message("\nDEBUG SUMMARY:")
        log_message("1. LLMService methods called:")
        log_message("   - __init__")
        log_message("   - build_metric_chains")
        log_message("   - generate_recommendations")
        log_message("   - build_evaluator_chain")
        log_message("   - evaluate_recommendations")
        
        log_message("2. metrics_calculator functions called:")
        log_message("   - calculate_metrics_for_recommendations")
        log_message("   - add_metrics_to_results")
        
        log_message("\nBoth modules are actively used during execution and contribute to the final results")
        
    except Exception as e:
        log_message(f"ERROR during execution: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

if __name__ == "__main__":
    print(f"Starting debug session. Trace will be written to {LOG_FILE}")
    asyncio.run(run_simplified_test())
    print(f"Debug session completed. See {LOG_FILE} for details.") 