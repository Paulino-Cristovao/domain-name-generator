#!/usr/bin/env python3
"""Evaluation script for domain generation models using OpenAI judge"""
import argparse
import sys
import os
import asyncio
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from domain_generator.models.inference import DomainGenerator
from domain_generator.evaluation.openai_judge import EvaluationFramework
from domain_generator.utils.config import Config

def load_test_cases(test_file: str) -> list:
    """Load test cases from file"""
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        # Create default test cases
        return create_default_test_cases()
    
    with open(test_file, 'r') as f:
        return json.load(f)

def create_default_test_cases() -> list:
    """Create default test cases for evaluation"""
    return [
        {
            "business_description": "organic coffee shop in downtown area",
            "expected_quality": "high"
        },
        {
            "business_description": "AI-powered restaurant management platform for small businesses", 
            "expected_quality": "high"
        },
        {
            "business_description": "sustainable fashion e-commerce platform",
            "expected_quality": "high"
        },
        {
            "business_description": "tech startup building mobile apps",
            "expected_quality": "medium"
        },
        {
            "business_description": "consulting firm helping businesses grow",
            "expected_quality": "medium"
        },
        {
            "business_description": "thing that does stuff",
            "expected_quality": "low"
        },
        {
            "business_description": "blockchain-based decentralized autonomous organization utilizing smart contracts",
            "expected_quality": "low"
        }
    ]

async def evaluate_single_model(
    model_path: str,
    base_model_name: str,
    test_cases: list,
    config: Config,
    num_suggestions: int = 5
) -> dict:
    """Evaluate a single model"""
    
    print(f"📝 Loading model from: {model_path}")
    
    # Initialize model
    try:
        generator = DomainGenerator(model_path, base_model_name, config)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return {"error": f"Failed to load model: {e}"}
    
    print(f"🎯 Generating domain suggestions...")
    
    # Generate suggestions for test cases
    model_results = []
    for i, test_case in enumerate(test_cases):
        print(f"  Processing case {i+1}/{len(test_cases)}: {test_case['business_description'][:50]}...")
        
        try:
            suggestions = generator.generate_with_confidence(
                test_case["business_description"],
                num_suggestions=num_suggestions
            )
            
            model_results.append({
                "business_description": test_case["business_description"],
                "expected_quality": test_case.get("expected_quality", "unknown"),
                "suggestions": suggestions
            })
            
        except Exception as e:
            print(f"⚠️  Error generating for case {i+1}: {e}")
            model_results.append({
                "business_description": test_case["business_description"],
                "expected_quality": test_case.get("expected_quality", "unknown"),
                "suggestions": [],
                "error": str(e)
            })
    
    return model_results

async def main():
    parser = argparse.ArgumentParser(description='Evaluate domain generation model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--base-model', type=str, required=True,
                       help='Base model name (e.g., mistralai/Mistral-7B-Instruct-v0.2)')
    parser.add_argument('--test-cases', type=str, default=None,
                       help='Path to test cases JSON file')
    parser.add_argument('--output-dir', type=str, default="data/results",
                       help='Output directory for results')
    parser.add_argument('--num-suggestions', type=int, default=5,
                       help='Number of domain suggestions per test case')
    parser.add_argument('--batch-size', type=int, default=3,
                       help='Batch size for OpenAI evaluation')
    
    args = parser.parse_args()
    
    # Validate model path exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = Config()
    
    # Load test cases
    print("📋 Loading test cases...")
    if args.test_cases:
        test_cases = load_test_cases(args.test_cases)
    else:
        test_cases = create_default_test_cases()
        print("  Using default test cases")
    
    print(f"  Loaded {len(test_cases)} test cases")
    
    # Initialize evaluation framework
    print("🔧 Initializing OpenAI evaluation framework...")
    try:
        eval_framework = EvaluationFramework(config)
    except Exception as e:
        print(f"❌ Failed to initialize evaluation framework: {e}")
        print("Please check your OpenAI API key in .env file")
        sys.exit(1)
    
    # Extract model name from path
    model_name = os.path.basename(args.model_path.rstrip('/'))
    
    print(f"🚀 Starting evaluation of {model_name}...")
    
    # Generate domain suggestions
    model_results = await evaluate_single_model(
        args.model_path,
        args.base_model,
        test_cases,
        config,
        args.num_suggestions
    )
    
    if "error" in model_results:
        print(f"❌ Evaluation failed: {model_results['error']}")
        sys.exit(1)
    
    # Evaluate with OpenAI judge
    print(f"⚖️  Evaluating with OpenAI judge...")
    evaluation_results = await eval_framework.evaluate_model_output(
        model_name,
        model_results,
        args.batch_size
    )
    
    # Print results summary
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    if "error" in evaluation_results:
        print(f"❌ Evaluation error: {evaluation_results['error']}")
    else:
        print(f"Model: {evaluation_results['model_name']}")
        print(f"Judge: {evaluation_results['judge_model']}")
        print(f"Total evaluations: {evaluation_results['total_evaluations']}")
        print()
        
        overall = evaluation_results['overall_score']
        print(f"Overall Score: {overall['mean']:.2f} ± {overall['std']:.2f}")
        print(f"  Range: {overall['min']:.2f} - {overall['max']:.2f}")
        print(f"  Median: {overall['median']:.2f}")
        print()
        
        print("Criterion Scores:")
        for criterion, stats in evaluation_results['criterion_scores'].items():
            weight = stats['weight']
            print(f"  {criterion.title():15} ({weight*100:2.0f}%): {stats['mean']:.2f} ± {stats['std']:.2f}")
        
        print("\nSample Evaluations:")
        for i, eval_sample in enumerate(evaluation_results.get('sample_evaluations', [])[:3]):
            print(f"\n  Example {i+1}:")
            print(f"    Business: {eval_sample['business_description'][:60]}...")
            print(f"    Domain: {eval_sample['domain']}")
            print(f"    Score: {eval_sample['overall_score']:.1f}")
            print(f"    Feedback: {eval_sample['feedback'][:100]}...")
    
    # Save results
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{args.output_dir}/{model_name}_evaluation_{timestamp}.json"
    
    # Combine model results and evaluation
    full_results = {
        "model_name": model_name,
        "model_path": args.model_path,
        "base_model": args.base_model,
        "test_cases": test_cases,
        "model_results": model_results,
        "evaluation": evaluation_results,
        "evaluation_timestamp": timestamp
    }
    
    eval_framework.save_evaluation_results(full_results, results_file)
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {results_file}")
    
    # Performance assessment
    if "overall_score" in evaluation_results:
        score = evaluation_results['overall_score']['mean']
        if score >= 8.0:
            print("🎉 Excellent performance! Model meets production targets.")
        elif score >= 7.0:
            print("👍 Good performance. Consider minor improvements.")
        elif score >= 6.0:
            print("⚠️  Moderate performance. Improvements recommended.")
        else:
            print("🔧 Poor performance. Significant improvements needed.")

if __name__ == "__main__":
    asyncio.run(main())