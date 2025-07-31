#!/usr/bin/env python3
"""
Compare baseline vs fine-tuned model performance for Phi-3-mini and Llama-3.2-1B
"""

import sys
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import torch
from tqdm.auto import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from domain_generator.models.inference import DomainGenerator
from domain_generator.models.trainer import create_model_configs
from domain_generator.utils.config import Config

class BaselineGenerator:
    """Baseline model generator (no fine-tuning)"""
    
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model_name = model_name
        print(f"📥 Loading baseline model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        self.model.eval()
        print("✅ Baseline model loaded")
    
    def generate_domains(self, business_description: str, num_suggestions: int = 5) -> List[str]:
        """Generate domain suggestions using baseline model"""
        prompt = f"Business: {business_description}\\nDomain suggestions:\\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = generated_text[len(prompt):]
        
        # Extract domains using regex
        import re
        domain_pattern = r'\\b[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\\.[a-z]{2,}\\b'
        domains = re.findall(domain_pattern, generated_part.lower())
        
        # Clean and filter
        unique_domains = []
        for domain in domains:
            if domain not in unique_domains and len(domain) > 4 and len(domain) < 50:
                unique_domains.append(domain)
        
        return unique_domains[:num_suggestions]

class ModelComparison:
    """Framework for comparing baseline vs fine-tuned models"""
    
    def __init__(self):
        self.model_configs = create_model_configs()
        self.test_cases = [
            "AI-powered fitness tracking app for runners",
            "sustainable coffee shop with co-working space", 
            "virtual reality gaming arcade for teenagers",
            "online language learning platform for professionals",
            "eco-friendly meal delivery service",
            "blockchain-based supply chain management system",
            "mental health meditation app for stressed workers",
            "artisanal chocolate subscription box service"
        ]
    
    def evaluate_baseline(self, model_name: str) -> List[Dict]:
        """Evaluate baseline model performance"""
        print(f"\\n🎯 Evaluating Baseline {self.model_configs[model_name]['display_name']}")
        print("=" * 60)
        
        baseline = BaselineGenerator(self.model_configs[model_name]["model_name"])
        results = []
        
        for i, test_case in enumerate(tqdm(self.test_cases, desc="Testing baseline"), 1):
            print(f"\\n{i}. {test_case}")
            
            start_time = time.time()
            try:
                domains = baseline.generate_domains(test_case, num_suggestions=5)
                generation_time = time.time() - start_time
                
                print(f"   ⏱️  Generated {len(domains)} domains in {generation_time:.2f}s")
                for j, domain in enumerate(domains[:3], 1):  # Show first 3
                    print(f"     {j}. {domain}")
                
                results.append({
                    'test_case': test_case,
                    'domains': domains,
                    'num_domains': len(domains),
                    'generation_time': generation_time,
                    'success': True
                })
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    'test_case': test_case,
                    'domains': [],
                    'num_domains': 0,
                    'generation_time': 0,
                    'success': False
                })
        
        # Clear memory
        del baseline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def evaluate_finetuned(self, model_name: str, model_path: str) -> List[Dict]:
        """Evaluate fine-tuned model performance"""
        print(f"\\n🎯 Evaluating Fine-tuned {self.model_configs[model_name]['display_name']}")
        print("=" * 60)
        
        config = Config()
        config.model.model_name = self.model_configs[model_name]["model_name"]
        
        generator = DomainGenerator(
            model_path=model_path,
            base_model_name=config.model.model_name,
            config=config
        )
        
        results = []
        
        for i, test_case in enumerate(tqdm(self.test_cases, desc="Testing fine-tuned"), 1):
            print(f"\\n{i}. {test_case}")
            
            start_time = time.time()
            try:
                domains = generator.generate_domains(test_case, num_suggestions=5)
                generation_time = time.time() - start_time
                
                print(f"   ⏱️  Generated {len(domains)} domains in {generation_time:.2f}s")
                for j, domain in enumerate(domains[:3], 1):  # Show first 3
                    print(f"     {j}. {domain}")
                
                results.append({
                    'test_case': test_case,
                    'domains': domains,
                    'num_domains': len(domains),
                    'generation_time': generation_time,
                    'success': True
                })
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    'test_case': test_case,
                    'domains': [],
                    'num_domains': 0,
                    'generation_time': 0,
                    'success': False
                })
        
        # Clear memory
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def compare_models(self, model_name: str, model_path: str = None) -> Dict:
        """Compare baseline vs fine-tuned performance"""
        print(f"\\n🔬 Comprehensive Model Comparison: {self.model_configs[model_name]['display_name']}")
        print("=" * 80)
        
        # Evaluate baseline
        baseline_results = self.evaluate_baseline(model_name)
        
        # Evaluate fine-tuned if path provided
        finetuned_results = []
        if model_path and os.path.exists(model_path):
            finetuned_results = self.evaluate_finetuned(model_name, model_path)
        else:
            print(f"\\n⚠️  Fine-tuned model not found at: {model_path}")
        
        # Calculate metrics
        comparison = {
            'model_name': model_name,
            'baseline': baseline_results,
            'finetuned': finetuned_results,
            'metrics': self._calculate_metrics(baseline_results, finetuned_results)
        }
        
        return comparison
    
    def _calculate_metrics(self, baseline_results: List[Dict], finetuned_results: List[Dict]) -> Dict:
        """Calculate comparison metrics"""
        def calc_stats(results):
            successful = [r for r in results if r['success']]
            if not successful:
                return {'avg_time': 0, 'avg_domains': 0, 'success_rate': 0, 'total_domains': 0}
            
            return {
                'avg_time': sum(r['generation_time'] for r in successful) / len(successful),
                'avg_domains': sum(r['num_domains'] for r in successful) / len(successful),
                'success_rate': len(successful) / len(results),
                'total_domains': sum(r['num_domains'] for r in successful)
            }
        
        baseline_stats = calc_stats(baseline_results)
        finetuned_stats = calc_stats(finetuned_results) if finetuned_results else {'avg_time': 0, 'avg_domains': 0, 'success_rate': 0, 'total_domains': 0}
        
        return {
            'baseline': baseline_stats,
            'finetuned': finetuned_stats,
            'improvement': {
                'time_change': finetuned_stats['avg_time'] - baseline_stats['avg_time'],
                'domain_change': finetuned_stats['avg_domains'] - baseline_stats['avg_domains'],
                'success_improvement': finetuned_stats['success_rate'] - baseline_stats['success_rate']
            }
        }
    
    def visualize_comparison(self, comparison: Dict):
        """Create visualizations for model comparison"""
        metrics = comparison['metrics']
        model_name = comparison['model_name']
        
        if not comparison['finetuned']:
            print("⚠️  No fine-tuned results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.model_configs[model_name]["display_name"]} - Baseline vs Fine-tuned Comparison', 
                     fontsize=16, fontweight='bold')
        
        # 1. Average generation time
        models = ['Baseline', 'Fine-tuned']
        times = [metrics['baseline']['avg_time'], metrics['finetuned']['avg_time']]
        colors = ['lightcoral', 'lightblue']
        
        axes[0, 0].bar(models, times, color=colors)
        axes[0, 0].set_title('Average Generation Time')
        axes[0, 0].set_ylabel('Time (seconds)')
        for i, v in enumerate(times):
            axes[0, 0].text(i, v + 0.01, f'{v:.2f}s', ha='center', va='bottom')
        
        # 2. Average domains generated
        domains = [metrics['baseline']['avg_domains'], metrics['finetuned']['avg_domains']]
        axes[0, 1].bar(models, domains, color=colors)
        axes[0, 1].set_title('Average Domains Generated')
        axes[0, 1].set_ylabel('Number of Domains')
        for i, v in enumerate(domains):
            axes[0, 1].text(i, v + 0.05, f'{v:.1f}', ha='center', va='bottom')
        
        # 3. Success rate
        success_rates = [metrics['baseline']['success_rate'] * 100, 
                        metrics['finetuned']['success_rate'] * 100]
        axes[1, 0].bar(models, success_rates, color=colors)
        axes[1, 0].set_title('Success Rate')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_ylim(0, 100)
        for i, v in enumerate(success_rates):
            axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # 4. Total domains generated
        total_domains = [metrics['baseline']['total_domains'], metrics['finetuned']['total_domains']]
        axes[1, 1].bar(models, total_domains, color=colors)
        axes[1, 1].set_title('Total Domains Generated')
        axes[1, 1].set_ylabel('Total Domains')
        for i, v in enumerate(total_domains):
            axes[1, 1].text(i, v + 0.5, f'{v}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'comparison_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\\n📊 {self.model_configs[model_name]['display_name']} Comparison Summary:")
        print(f"  📈 Avg Generation Time: {metrics['improvement']['time_change']:+.2f}s")
        print(f"  📈 Avg Domains: {metrics['improvement']['domain_change']:+.1f}")
        print(f"  📈 Success Rate: {metrics['improvement']['success_improvement']*100:+.1f}%")

def main():
    """Main comparison function"""
    print("🚀 Domain Name Generator: Baseline vs Fine-tuned Comparison")
    print("=" * 80)
    
    comparator = ModelComparison()
    
    # Define model paths (update these paths to your trained models)
    model_paths = {
        "llama-3.2-1b": "models/llama-3.2-1b-domain-generator/final",
        "phi-3-mini": "models/phi-3-mini-domain-generator/final"
    }
    
    results = {}
    
    # Compare both models
    for model_name, model_path in model_paths.items():
        try:
            print(f"\\n🔍 Comparing {model_name}...")
            comparison = comparator.compare_models(model_name, model_path)
            results[model_name] = comparison
            
            # Visualize if fine-tuned model exists
            if comparison['finetuned']:
                comparator.visualize_comparison(comparison)
                
        except Exception as e:
            print(f"❌ Failed to compare {model_name}: {e}")
            continue
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data/results/baseline_vs_finetuned_comparison_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n📄 Results saved to: {results_file}")
    
    # Create summary table
    if results:
        summary_data = []
        for model_name, comparison in results.items():
            if comparison['baseline']:
                baseline_metrics = comparison['metrics']['baseline']
                finetuned_metrics = comparison['metrics']['finetuned']
                
                summary_data.extend([
                    {
                        'model': comparator.model_configs[model_name]['display_name'],
                        'type': 'baseline',
                        'avg_time': baseline_metrics['avg_time'],
                        'avg_domains': baseline_metrics['avg_domains'],
                        'success_rate': baseline_metrics['success_rate'],
                        'total_domains': baseline_metrics['total_domains']
                    },
                    {
                        'model': comparator.model_configs[model_name]['display_name'],
                        'type': 'fine-tuned',
                        'avg_time': finetuned_metrics['avg_time'],
                        'avg_domains': finetuned_metrics['avg_domains'],
                        'success_rate': finetuned_metrics['success_rate'],
                        'total_domains': finetuned_metrics['total_domains']
                    }
                ])
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_file = f"data/results/baseline_vs_finetuned_summary_{timestamp}.csv"
            summary_df.to_csv(csv_file, index=False)
            
            print(f"📊 Summary saved to: {csv_file}")
            print("\\n📋 Performance Summary:")
            print(summary_df.round(3))
    
    print("\\n🎉 Comparison complete!")

if __name__ == "__main__":
    main()