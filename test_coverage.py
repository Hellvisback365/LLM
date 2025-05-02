import os
import sys
import json
import unittest
import coverage
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

# Add the project root to path if needed
sys.path.append('.')

# Setup coverage
cov = coverage.Coverage(
    source=['src.recommender.api.llm_service', 'src.recommender.core.metrics_calculator'],
    omit=['*/__init__.py', '*/test_*.py']
)
cov.start()

# Import modules under test
from src.recommender.api.llm_service import LLMService
from src.recommender.core.metrics_calculator import calculate_metrics_for_recommendations, add_metrics_to_results
from src.recommender.utils.rag_utils import calculate_coverage

class TestLLMService(unittest.TestCase):
    """Tests for LLMService class"""
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "mock_key"})
    def setUp(self):
        self.llm_service = LLMService(model_name="test_model", temperature=0.5)
    
    def test_initialization(self):
        self.assertEqual(self.llm_service.model_name, "test_model")
        self.assertEqual(self.llm_service.common_params["temperature"], 0.5)
        self.assertEqual(self.llm_service.common_params["openai_api_key"], "mock_key")
    
    @patch('src.recommender.api.llm_service.ChatOpenAI')
    def test_create_llm(self, mock_chat_openai):
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        result = self.llm_service._create_llm()
        
        mock_chat_openai.assert_called_once_with(
            model="test_model", 
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key="mock_key",
            temperature=0.5,
            max_tokens=512
        )
        self.assertEqual(result, mock_instance)
    
    def test_make_prompt_with_parser(self):
        metric_key = "test_metric"
        metric_text = "Test metric description"
        example = "[1, 2, 3]"
        
        prompt_template, parser = self.llm_service.make_prompt_with_parser(
            metric_key, metric_text, example
        )
        
        self.assertIn(metric_key, prompt_template.template)
        self.assertIn(metric_text, prompt_template.template)
        self.assertIn(example, prompt_template.template)
        self.assertEqual(len(prompt_template.input_variables), 2)
        self.assertIn("catalog", prompt_template.input_variables)
        self.assertIn("user_profile", prompt_template.input_variables)
    
    @patch('src.recommender.api.llm_service.LLMService._create_llm')
    def test_build_metric_chains(self, mock_create_llm):
        mock_create_llm.return_value = MagicMock()
        
        metrics_definitions = {
            "accuracy": "Test accuracy description",
            "diversity": "Test diversity description"
        }
        
        chains, parsers, raw_prompts = self.llm_service.build_metric_chains(metrics_definitions)
        
        self.assertEqual(len(chains), 2)
        self.assertEqual(len(parsers), 2)
        self.assertEqual(len(raw_prompts), 2)
        self.assertIn("accuracy", chains)
        self.assertIn("diversity", chains)
        self.assertIn("accuracy", parsers)
        self.assertIn("diversity", parsers)
        self.assertIn("accuracy", raw_prompts)
        self.assertIn("diversity", raw_prompts)
    
    @patch('src.recommender.api.llm_service.LLMService._create_llm')
    def test_build_evaluator_chain(self, mock_create_llm):
        mock_create_llm.return_value = MagicMock()
        
        chain, parser = self.llm_service.build_evaluator_chain()
        
        self.assertIsNotNone(chain)
        self.assertIsNotNone(parser)

class TestCoverage(unittest.TestCase):
    """Tests for coverage calculation"""
    
    def setUp(self):
        # Mock movie data
        self.movies_data = [
            {"movie_id": 1, "title": "Movie 1", "genres": "Action|Adventure"},
            {"movie_id": 2, "title": "Movie 2", "genres": "Comedy|Drama"},
            {"movie_id": 3, "title": "Movie 3", "genres": "Horror|Thriller"},
            {"movie_id": 4, "title": "Movie 4", "genres": "Action|Sci-Fi"},
            {"movie_id": 5, "title": "Movie 5", "genres": "Comedy|Romance"}
        ]
        self.movies_df = pd.DataFrame(self.movies_data)
    
    def test_calculate_coverage_empty_input(self):
        """Test coverage calculation with empty input"""
        result = calculate_coverage([], self.movies_df)
        self.assertEqual(result["total_coverage"], 0.0)
        self.assertEqual(result["genre_coverage"], 0.0)
    
    def test_calculate_coverage_single_movie(self):
        """Test coverage calculation with a single movie"""
        result = calculate_coverage([1], self.movies_df)
        self.assertEqual(result["total_coverage"], 0.2)  # 1/5
        self.assertGreater(result["genre_coverage"], 0.0)
    
    def test_calculate_coverage_all_movies(self):
        """Test coverage calculation with all movies"""
        result = calculate_coverage([1, 2, 3, 4, 5], self.movies_df)
        self.assertEqual(result["total_coverage"], 1.0)
        self.assertEqual(result["genre_coverage"], 1.0)
    
    def test_calculate_coverage_genre_diversity(self):
        """Test coverage calculation focusing on genre diversity"""
        # Seleziona film con generi diversi
        result = calculate_coverage([1, 2, 3], self.movies_df)
        self.assertEqual(result["total_coverage"], 0.6)  # 3/5
        self.assertGreater(result["genre_coverage"], 0.5)  # Dovrebbe coprire la maggior parte dei generi

class TestMetricsCalculator(unittest.TestCase):
    """Tests for metrics_calculator functions"""
    
    def setUp(self):
        # Mock data
        self.metric_results = {
            "precision_at_k": {
                "recommendations": [1, 2, 3],
                "output": "Test output"
            },
            "coverage": {
                "recommendations": [4, 5, 6],
                "output": "Test output"
            }
        }
        self.final_evaluation = {
            "final_recommendations": [1, 5, 9],
            "justification": "Test justification"
        }
        
        # Mock catalog data
        self.mock_movies_data = [
            {"movie_id": 1, "title": "Test Movie 1", "genres": "Comedy|Drama"},
            {"movie_id": 2, "title": "Test Movie 2", "genres": "Action|Adventure"},
            {"movie_id": 3, "title": "Test Movie 3", "genres": "Horror|Thriller"},
            {"movie_id": 4, "title": "Test Movie 4", "genres": "Romance|Drama"},
            {"movie_id": 5, "title": "Test Movie 5", "genres": "Sci-Fi|Action"},
            {"movie_id": 6, "title": "Test Movie 6", "genres": "Documentary"},
            {"movie_id": 9, "title": "Test Movie 9", "genres": "Comedy|Romance"}
        ]
    
    @patch('os.path.join')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_calculate_metrics_for_recommendations(self, mock_json_load, mock_open, mock_path_join):
        # Setup mocks
        mock_path_join.return_value = 'mock_path'
        mock_json_load.return_value = self.mock_movies_data
        
        # Run the function
        result = calculate_metrics_for_recommendations(self.metric_results, self.final_evaluation)
        
        # Assert result structure
        self.assertIsInstance(result, dict)
        self.assertIn('precision_at_k', result)
        self.assertIn('coverage', result)
        self.assertIn('final_recommendations', result)
        self.assertIn('system_coverage', result)
        
        # Check metrics structure
        for metric in ['precision_at_k', 'coverage', 'final_recommendations']:
            self.assertIn('precision_score', result[metric])
            self.assertIn('total_coverage', result[metric])
            self.assertIn('genre_coverage', result[metric])
        
        # Check system coverage
        self.assertIn('total_coverage', result['system_coverage'])
        self.assertIn('genre_coverage', result['system_coverage'])
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('json.dump')
    def test_add_metrics_to_results(self, mock_json_dump, mock_json_load, mock_open, mock_path_exists):
        # Setup mocks
        mock_path_exists.return_value = True
        mock_json_load.return_value = {
            "timestamp": "2023-01-01T00:00:00",
            "metric_results": self.metric_results,
            "final_evaluation": self.final_evaluation
        }
        
        # Mock metrics to add
        metrics = {
            "precision_at_k": {
                "precision_score": 0.5,
                "total_coverage": 0.3,
                "genre_coverage": 0.6
            },
            "coverage": {
                "precision_score": 0.4,
                "total_coverage": 0.5,
                "genre_coverage": 0.7
            },
            "final_recommendations": {
                "precision_score": 0.6,
                "total_coverage": 0.4,
                "genre_coverage": 0.5
            },
            "system_coverage": {
                "total_coverage": 0.6,
                "genre_coverage": 0.8
            }
        }
        
        # Run the function
        result = add_metrics_to_results(metrics, "test_output.json")
        
        # Assertions
        self.assertTrue(result)
        mock_path_exists.assert_called_once_with("test_output.json")
        mock_open.assert_any_call("test_output.json", "r", encoding="utf-8")
        mock_open.assert_any_call("test_output.json", "w", encoding="utf-8")
        mock_json_load.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # Get the actual data passed to json.dump
        actual_data = mock_json_dump.call_args[0][0]
        self.assertIn("metrics", actual_data)
        self.assertEqual(actual_data["metrics"], metrics)

if __name__ == "__main__":
    # Run the unittest tests
    print("Running tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Stop coverage and report
    cov.stop()
    cov.save()
    
    print("\nCoverage Report:")
    cov.report()
    
    # Generate HTML report
    cov_dir = 'htmlcov'
    os.makedirs(cov_dir, exist_ok=True)
    cov.html_report(directory=cov_dir)
    print(f"\nHTML coverage report generated in {cov_dir}/") 