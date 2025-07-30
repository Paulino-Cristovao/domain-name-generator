"""LLM-as-a-Judge evaluation framework for domain name quality assessment"""
import os
import json
import time
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import openai
import anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

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

class LLMJudge:
    """Base class for LLM-based domain evaluation"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.criteria = config.criteria
        
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
1. A score from 1-10 for each criterion
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
            "relevance": "How well does the domain relate to the business description?",
            "memorability": "Is the domain easy to remember, spell, and type?",
            "professionalism": "Does the domain sound credible and trustworthy?",
            "length": "Is the domain an appropriate length (6-15 characters is ideal)?",
            "clarity": "Is the meaning/purpose immediately clear from the domain?"
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
        """Evaluate a single domain (to be implemented by subclasses)"""
        raise NotImplementedError
    
    async def evaluate_domains_batch(
        self, 
        evaluations: List[Tuple[str, str]]
    ) -> List[DomainEvaluation]:
        """Evaluate multiple domains in batch"""
        
        tasks = []
        for business_desc, domain in evaluations:
            task = self.evaluate_domain(business_desc, domain)
            tasks.append(task)
        
        # Process with rate limiting
        results = []
        batch_size = 5  # Process 5 at a time to avoid rate limits
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"Error in evaluation: {result}")
                else:
                    results.append(result)
            
            # Rate limiting delay
            if i + batch_size < len(tasks):
                await asyncio.sleep(1)
        
        return results

class OpenAIJudge(LLMJudge):
    """OpenAI GPT-based domain evaluator"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        
        # Initialize OpenAI client
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {config.api_key_env}")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
    async def evaluate_domain(self, business_description: str, domain: str) -> DomainEvaluation:
        """Evaluate domain using OpenAI GPT"""
        
        prompt = self.create_evaluation_prompt(business_description, domain)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert domain name evaluator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            evaluation_data = json.loads(content)
            
            # Validate and calculate overall score
            scores = evaluation_data["scores"]
            overall_score = self.calculate_overall_score(scores)
            
            return DomainEvaluation(
                domain=domain,
                business_description=business_description,
                scores=scores,
                overall_score=overall_score,
                feedback=evaluation_data["feedback"],
                judge_model=self.config.judge_model
            )
            
        except Exception as e:
            print(f"Error evaluating domain {domain}: {e}")
            
            # Return default evaluation on error
            return DomainEvaluation(
                domain=domain,
                business_description=business_description,
                scores={k: 5.0 for k in self.criteria.keys()},
                overall_score=5.0,
                feedback=f"Evaluation failed: {str(e)}",
                judge_model=self.config.judge_model
            )

class AnthropicJudge(LLMJudge):
    """Anthropic Claude-based domain evaluator"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        
        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        
    async def evaluate_domain(self, business_description: str, domain: str) -> DomainEvaluation:
        """Evaluate domain using Anthropic Claude"""
        
        prompt = self.create_evaluation_prompt(business_description, domain)
        
        try:
            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            
            # Parse JSON response
            evaluation_data = json.loads(content)
            
            # Validate and calculate overall score
            scores = evaluation_data["scores"]
            overall_score = self.calculate_overall_score(scores)
            
            return DomainEvaluation(
                domain=domain,
                business_description=business_description,
                scores=scores,
                overall_score=overall_score,
                feedback=evaluation_data["feedback"],
                judge_model="claude-3-sonnet"
            )
            
        except Exception as e:
            print(f"Error evaluating domain {domain}: {e}")
            
            # Return default evaluation on error
            return DomainEvaluation(
                domain=domain,
                business_description=business_description,
                scores={k: 5.0 for k in self.criteria.keys()},
                overall_score=5.0,
                feedback=f"Evaluation failed: {str(e)}",
                judge_model="claude-3-sonnet"
            )

class EvaluationFramework:
    """Complete evaluation framework for domain generation models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.judges = {}
        
        # Initialize available judges
        try:
            self.judges["openai"] = OpenAIJudge(config.evaluation)
        except ValueError as e:
            print(f"OpenAI judge not available: {e}")
        
        try:
            self.judges["anthropic"] = AnthropicJudge(config.evaluation)
        except ValueError as e:
            print(f"Anthropic judge not available: {e}")
        
        if not self.judges:
            raise ValueError("No LLM judges available. Please set up API keys.")
    
    async def evaluate_model_output(
        self,
        model_name: str,
        test_cases: List[Dict],
        judge_name: str = "openai"
    ) -> Dict[str, any]:
        """Evaluate a model's domain suggestions on test cases"""
        
        if judge_name not in self.judges:
            raise ValueError(f"Judge {judge_name} not available")
        
        judge = self.judges[judge_name]
        
        # Prepare evaluation tasks
        evaluation_tasks = []
        for case in test_cases:
            business_desc = case["business_description"]
            for suggestion in case.get("suggestions", []):
                domain = suggestion.get("domain", suggestion) if isinstance(suggestion, dict) else suggestion
                evaluation_tasks.append((business_desc, domain))
        
        print(f"Evaluating {len(evaluation_tasks)} domain suggestions with {judge_name} judge...")
        
        # Run evaluations
        evaluations = await judge.evaluate_domains_batch(evaluation_tasks)
        
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
            "overall_score": {
                "mean": np.mean(overall_scores),
                "std": np.std(overall_scores),
                "median": np.median(overall_scores),
                "min": np.min(overall_scores),
                "max": np.max(overall_scores)
            },
            "criterion_scores": {}
        }
        
        for criterion, scores in all_scores.items():
            if scores:
                aggregate_stats["criterion_scores"][criterion] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "median": np.median(scores)
                }
        
        # Add detailed evaluations
        aggregate_stats["detailed_evaluations"] = [eval.to_dict() for eval in evaluations]
        
        return aggregate_stats
    
    async def compare_models(
        self,
        model_results: Dict[str, List[Dict]],
        judge_name: str = "openai"
    ) -> Dict[str, any]:
        """Compare multiple models using LLM judge"""
        
        comparison_results = {}
        
        for model_name, test_cases in model_results.items():
            print(f"Evaluating model: {model_name}")
            
            model_evaluation = await self.evaluate_model_output(
                model_name,
                test_cases,
                judge_name
            )
            
            comparison_results[model_name] = model_evaluation
        
        # Add comparative analysis
        comparison_results["comparison"] = self._create_model_comparison(comparison_results)
        
        return comparison_results
    
    def _create_model_comparison(self, results: Dict[str, any]) -> Dict[str, any]:
        """Create comparative analysis between models"""
        
        model_names = [name for name in results.keys() if name != "comparison"]
        
        if len(model_names) < 2:
            return {"error": "Need at least 2 models for comparison"}
        
        comparison = {
            "model_ranking": [],
            "statistical_significance": {},
            "best_criteria": {}
        }
        
        # Rank models by overall score
        model_scores = []
        for model_name in model_names:
            mean_score = results[model_name]["overall_score"]["mean"]
            model_scores.append((model_name, mean_score))
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        comparison["model_ranking"] = model_scores
        
        # Find best model for each criterion
        for criterion in self.config.evaluation.criteria.keys():
            best_score = 0
            best_model = None
            
            for model_name in model_names:
                if criterion in results[model_name]["criterion_scores"]:
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
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_json = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Evaluation results saved to: {output_path}")

# Utility functions for validation
def validate_judge_availability() -> Dict[str, bool]:
    """Check which LLM judges are available"""
    availability = {}
    
    # Check OpenAI
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        availability["openai"] = bool(api_key)
    except:
        availability["openai"] = False
    
    # Check Anthropic
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        availability["anthropic"] = bool(api_key)
    except:
        availability["anthropic"] = False
    
    return availability

async def test_judge_setup(config: Config) -> bool:
    """Test if LLM judge setup is working"""
    try:
        framework = EvaluationFramework(config)
        
        # Simple test evaluation
        test_business = "organic coffee shop"
        test_domain = "organicbeans.com"
        
        judge_name = list(framework.judges.keys())[0]
        judge = framework.judges[judge_name]
        
        result = await judge.evaluate_domain(test_business, test_domain)
        
        print(f"Test evaluation successful with {judge_name}")
        print(f"Test result: {result.overall_score}")
        
        return True
        
    except Exception as e:
        print(f"Judge setup test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the evaluation framework
    config = Config()
    
    # Check availability
    availability = validate_judge_availability()
    print("Judge availability:", availability)
    
    # Test setup
    async def main():
        success = await test_judge_setup(config)
        if success:
            print("LLM judge setup is working correctly!")
        else:
            print("Please check your API key configuration.")
    
    asyncio.run(main())