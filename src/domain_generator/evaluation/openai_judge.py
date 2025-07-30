"""OpenAI-only LLM-as-a-Judge evaluation framework"""
import os
import json
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import openai
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ..utils.config import Config, EvaluationConfig

@dataclass
class DomainEvaluation:
    """Structure for domain evaluation results"""
    domain: str
    business_description: str
    scores: Dict[str, float]
    overall_score: float
    feedback: str
    judge_model: str
    
    def to_dict(self) -> Dict:
        return {
            "domain": self.domain,
            "business_description": self.business_description,
            "scores": self.scores,
            "overall_score": self.overall_score,
            "feedback": self.feedback,
            "judge_model": self.judge_model
        }

class OpenAIJudge:
    """OpenAI GPT-based domain evaluator"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.criteria = config.criteria
        
        # Initialize OpenAI client
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {config.api_key_env}")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        print(f"✓ OpenAI judge initialized with model: {config.judge_model}")
        
    def create_evaluation_prompt(self, business_description: str, domain: str) -> str:
        """Create evaluation prompt for domain assessment"""
        
        criteria_desc = []
        for criterion, weight in self.criteria.items():
            criteria_desc.append(f"- {criterion.title()} ({weight*100:.0f}%): {self._get_criterion_description(criterion)}")
        
        criteria_text = "\n".join(criteria_desc)
        
        prompt = f"""You are an expert domain name evaluator. Please evaluate the following domain suggestion for a business.

Business Description: {business_description}
Proposed Domain: {domain}

Evaluation Criteria:
{criteria_text}

Please provide:
1. A score from 1-10 for each criterion (10 being excellent, 1 being poor)
2. An overall weighted score (1-10)
3. Brief feedback explaining your evaluation

Format your response as JSON:
{{
    "scores": {{
        "relevance": <score>,
        "memorability": <score>,
        "professionalism": <score>,
        "length": <score>,
        "clarity": <score>
    }},
    "overall_score": <weighted_average>,
    "feedback": "Brief explanation of the evaluation"
}}

Be objective and consistent in your scoring. Consider real-world usability and brand potential."""
        
        return prompt
    
    def _get_criterion_description(self, criterion: str) -> str:
        """Get description for each evaluation criterion"""
        descriptions = {
            "relevance": "How well does the domain relate to the business description and industry?",
            "memorability": "Is the domain easy to remember, spell, and type without errors?",
            "professionalism": "Does the domain sound credible, trustworthy, and professional?",
            "length": "Is the domain an appropriate length? (6-15 characters is ideal)",
            "clarity": "Is the meaning/purpose immediately clear from the domain name?"
        }
        return descriptions.get(criterion, "Evaluate this aspect of the domain")
    
    def calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        total_score = 0.0
        for criterion, score in scores.items():
            weight = self.criteria.get(criterion, 0.0)
            total_score += score * weight
        
        return round(total_score, 2)
    
    async def evaluate_domain(self, business_description: str, domain: str) -> DomainEvaluation:
        """Evaluate a single domain using OpenAI GPT"""
        
        prompt = self.create_evaluation_prompt(business_description, domain)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert domain name evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up JSON response (remove any markdown formatting)
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON response
            evaluation_data = json.loads(content)
            
            # Validate scores
            scores = evaluation_data["scores"]
            for criterion in self.criteria.keys():
                if criterion not in scores:
                    scores[criterion] = 5.0  # Default score
                else:
                    # Ensure score is within valid range
                    scores[criterion] = max(1.0, min(10.0, float(scores[criterion])))
            
            # Calculate overall score
            overall_score = self.calculate_overall_score(scores)
            
            return DomainEvaluation(
                domain=domain,
                business_description=business_description,
                scores=scores,
                overall_score=overall_score,
                feedback=evaluation_data.get("feedback", "No feedback provided"),
                judge_model=self.config.judge_model
            )
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for domain {domain}: {e}")
            print(f"Response content: {content}")
            
            # Return default evaluation on JSON error
            return self._create_default_evaluation(domain, business_description, f"JSON parsing failed: {str(e)}")
            
        except Exception as e:
            print(f"Error evaluating domain {domain}: {e}")
            
            # Return default evaluation on other errors
            return self._create_default_evaluation(domain, business_description, f"Evaluation failed: {str(e)}")
    
    def _create_default_evaluation(self, domain: str, business_description: str, error_msg: str) -> DomainEvaluation:
        """Create a default evaluation when API call fails"""
        return DomainEvaluation(
            domain=domain,
            business_description=business_description,
            scores={k: 5.0 for k in self.criteria.keys()},
            overall_score=5.0,
            feedback=error_msg,
            judge_model=self.config.judge_model
        )
    
    async def evaluate_domains_batch(
        self, 
        evaluations: List[Tuple[str, str]],
        batch_size: int = 3
    ) -> List[DomainEvaluation]:
        """Evaluate multiple domains in batches with rate limiting"""
        
        tasks = []
        for business_desc, domain in evaluations:
            task = self.evaluate_domain(business_desc, domain)
            tasks.append(task)
        
        results = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1} ({len(batch)} evaluations)...")
            
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"Error in batch evaluation: {result}")
                else:
                    results.append(result)
            
            # Rate limiting delay between batches
            if i + batch_size < len(tasks):
                await asyncio.sleep(2)  # 2 second delay between batches
        
        return results

class EvaluationFramework:
    """Complete evaluation framework using OpenAI judge"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize OpenAI judge
        try:
            self.judge = OpenAIJudge(config.evaluation)
            print("✓ Evaluation framework initialized with OpenAI judge")
        except ValueError as e:
            raise ValueError(f"Failed to initialize OpenAI judge: {e}")
    
    async def evaluate_model_output(
        self,
        model_name: str,
        test_cases: List[Dict],
        batch_size: int = 3
    ) -> Dict[str, any]:
        """Evaluate a model's domain suggestions on test cases"""
        
        # Prepare evaluation tasks
        evaluation_tasks = []
        for case in test_cases:
            business_desc = case["business_description"]
            suggestions = case.get("suggestions", [])
            
            if isinstance(suggestions, list):
                for suggestion in suggestions:
                    domain = suggestion.get("domain", suggestion) if isinstance(suggestion, dict) else suggestion
                    evaluation_tasks.append((business_desc, domain))
            else:
                print(f"Warning: No suggestions found for case: {business_desc}")
        
        if not evaluation_tasks:
            return {"error": "No evaluation tasks found"}
        
        print(f"Evaluating {len(evaluation_tasks)} domain suggestions for {model_name}...")
        
        # Run evaluations
        evaluations = await self.judge.evaluate_domains_batch(evaluation_tasks, batch_size)
        
        # Aggregate results
        results = self._aggregate_evaluation_results(evaluations, model_name, test_cases)
        
        return results
    
    def _aggregate_evaluation_results(
        self,
        evaluations: List[DomainEvaluation],
        model_name: str,
        test_cases: List[Dict]
    ) -> Dict[str, any]:
        """Aggregate evaluation results into summary statistics"""
        
        if not evaluations:
            return {"error": "No evaluations completed"}
        
        # Calculate aggregate scores
        all_scores = {criterion: [] for criterion in self.config.evaluation.criteria.keys()}
        overall_scores = []
        
        for eval_result in evaluations:
            overall_scores.append(eval_result.overall_score)
            for criterion, score in eval_result.scores.items():
                if criterion in all_scores:
                    all_scores[criterion].append(score)
        
        # Calculate statistics
        aggregate_stats = {
            "model_name": model_name,
            "total_evaluations": len(evaluations),
            "judge_model": self.config.evaluation.judge_model,
            "overall_score": {
                "mean": float(np.mean(overall_scores)),
                "std": float(np.std(overall_scores)),
                "median": float(np.median(overall_scores)),
                "min": float(np.min(overall_scores)),
                "max": float(np.max(overall_scores))
            },
            "criterion_scores": {}
        }
        
        for criterion, scores in all_scores.items():
            if scores:
                aggregate_stats["criterion_scores"][criterion] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "median": float(np.median(scores)),
                    "weight": self.config.evaluation.criteria[criterion]
                }
        
        # Add detailed evaluations (sample of 10)
        sample_evaluations = evaluations[:10] if len(evaluations) > 10 else evaluations
        aggregate_stats["sample_evaluations"] = [eval.to_dict() for eval in sample_evaluations]
        
        return aggregate_stats
    
    async def compare_models(
        self,
        model_results: Dict[str, List[Dict]],
        batch_size: int = 3
    ) -> Dict[str, any]:
        """Compare multiple models using OpenAI judge"""
        
        comparison_results = {}
        
        for model_name, test_cases in model_results.items():
            print(f"\n📊 Evaluating model: {model_name}")
            
            model_evaluation = await self.evaluate_model_output(
                model_name,
                test_cases,
                batch_size
            )
            
            comparison_results[model_name] = model_evaluation
        
        # Add comparative analysis
        comparison_results["comparison"] = self._create_model_comparison(comparison_results)
        
        return comparison_results
    
    def _create_model_comparison(self, results: Dict[str, any]) -> Dict[str, any]:
        """Create comparative analysis between models"""
        
        model_names = [name for name in results.keys() if name != "comparison"]
        
        if len(model_names) < 2:
            return {"note": "Need at least 2 models for comparison"}
        
        comparison = {
            "model_ranking": [],
            "performance_summary": {},
            "best_criteria": {}
        }
        
        # Rank models by overall score
        model_scores = []
        for model_name in model_names:
            if "overall_score" in results[model_name]:
                mean_score = results[model_name]["overall_score"]["mean"]
                model_scores.append((model_name, mean_score))
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        comparison["model_ranking"] = [{"model": name, "score": score} for name, score in model_scores]
        
        # Performance summary
        for model_name in model_names:
            if "overall_score" in results[model_name]:
                comparison["performance_summary"][model_name] = {
                    "overall_mean": results[model_name]["overall_score"]["mean"],
                    "overall_std": results[model_name]["overall_score"]["std"],
                    "total_evaluations": results[model_name]["total_evaluations"]
                }
        
        # Find best model for each criterion
        for criterion in self.config.evaluation.criteria.keys():
            best_score = 0
            best_model = None
            
            for model_name in model_names:
                if ("criterion_scores" in results[model_name] and 
                    criterion in results[model_name]["criterion_scores"]):
                    score = results[model_name]["criterion_scores"][criterion]["mean"]
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                comparison["best_criteria"][criterion] = {
                    "model": best_model,
                    "score": best_score
                }
        
        return comparison
    
    def save_evaluation_results(self, results: Dict, output_path: str):
        """Save evaluation results to file"""
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Evaluation results saved to: {output_path}")

# Utility functions
def test_openai_connection() -> bool:
    """Test OpenAI API connection"""
    try:
        config = Config()
        judge = OpenAIJudge(config.evaluation)
        
        # Simple test
        async def test():
            result = await judge.evaluate_domain("coffee shop", "testcafe.com")
            return result.overall_score > 0
        
        return asyncio.run(test())
        
    except Exception as e:
        print(f"OpenAI connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the evaluation framework
    print("🧪 Testing OpenAI evaluation framework...")
    
    if test_openai_connection():
        print("✅ OpenAI judge is working correctly!")
    else:
        print("❌ OpenAI judge setup failed. Please check your API key.")
        
    # Example usage
    async def example_usage():
        config = Config()
        framework = EvaluationFramework(config)
        
        # Test evaluation
        test_case = {
            "business_description": "innovative AI-powered coffee shop",
            "suggestions": ["aicafe.com", "smartbrew.co", "coffeebotai.net"]
        }
        
        results = await framework.evaluate_model_output("test_model", [test_case])
        
        print("\n📊 Example evaluation results:")
        print(f"Overall score: {results['overall_score']['mean']:.2f}")
        print(f"Total evaluations: {results['total_evaluations']}")
        
        return results
    
    # Run example (commented out to avoid API calls during import)
    # asyncio.run(example_usage())