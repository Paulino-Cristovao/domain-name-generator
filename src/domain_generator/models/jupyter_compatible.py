"""Jupyter-compatible wrapper for domain generation models"""
import os
import sys
from typing import List, Dict, Optional, Union
from pathlib import Path
import torch
from dotenv import load_dotenv
import wandb

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from domain_generator.models.trainer import DomainGeneratorTrainer, create_model_configs
from domain_generator.models.inference import DomainGenerator
from domain_generator.utils.config import Config

# Load environment variables
load_dotenv()

class JupyterDomainGenerator:
    """Jupyter-friendly wrapper for domain generation with same performance as CLI"""
    
    def __init__(self, model_name: str = "llama-3.2-1b") -> None:
        """Initialize the domain generator for Jupyter use.
        
        Args:
            model_name: Model configuration to use ('llama-3.2-1b', 'phi-3-mini', 'dialogpt-medium', 'gpt2-small', 'distilgpt2')
        """
        self.config = Config()
        self.model_name = model_name
        self.model_configs = create_model_configs()
        self.trainer: Optional[DomainGeneratorTrainer] = None
        self.generator: Optional[DomainGenerator] = None
        
        # Set up model configuration
        if model_name in self.model_configs:
            model_config = self.model_configs[model_name]
            self.config.model.model_name = model_config["model_name"]
            self.config.lora = model_config["lora_config"]
            self.config.training = model_config["training_config"]
        else:
            available_models = list(self.model_configs.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available_models}")
    
    def train_model(
        self, 
        dataset_path: str = "data/processed/training_dataset.json",
        output_dir: Optional[str] = None,
        use_wandb: bool = True
    ) -> str:
        """Train a domain generation model.
        
        Args:
            dataset_path: Path to training dataset
            output_dir: Output directory for model (auto-generated if None)
            use_wandb: Whether to use Weights & Biases logging
            
        Returns:
            Path to trained model
        """
        if output_dir is None:
            output_dir = f"models/{self.model_name}-domain-generator"
        
        # Initialize trainer
        self.trainer = DomainGeneratorTrainer(self.config)
        
        # Set up W&B if requested
        if use_wandb and os.getenv("WANDB_API_KEY"):
            print("✅ Weights & Biases logging enabled")
        elif use_wandb:
            print("⚠️  WANDB_API_KEY not found, skipping W&B logging")
        
        # Train model
        print(f"🚀 Starting training with {self.model_name}")
        print(f"📊 Model: {self.config.model.model_name}")
        print(f"💾 Output: {output_dir}")
        print(f"🔧 Device: {self.config.device}")
        
        model_path = self.trainer.train(
            dataset_path=dataset_path,
            output_dir=output_dir,
            model_name=self.config.model.model_name
        )
        
        print(f"✅ Training completed: {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model for inference.
        
        Args:
            model_path: Path to trained model directory
        """
        print(f"📥 Loading model from: {model_path}")
        
        self.generator = DomainGenerator(
            model_path=model_path,
            base_model_name=self.config.model.model_name,
            config=self.config
        )
        
        print("✅ Model loaded successfully")
    
    def generate_domains(
        self,
        business_description: str,
        target_audience: Optional[str] = None,
        num_suggestions: int = 5,
        temperature: float = 0.7,
        with_confidence: bool = True
    ) -> Union[List[str], List[Dict[str, float]]]:
        """Generate domain name suggestions.
        
        Args:
            business_description: Description of the business
            target_audience: Optional target audience description
            num_suggestions: Number of suggestions to generate
            temperature: Sampling temperature (0.1-1.0)
            with_confidence: Whether to return confidence scores
            
        Returns:
            List of domain suggestions or list of dicts with confidence scores
        """
        if self.generator is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if with_confidence:
            return self.generator.generate_with_confidence(
                business_description=business_description,
                target_audience=target_audience,
                num_suggestions=num_suggestions
            )
        else:
            return self.generator.generate_domains(
                business_description=business_description,
                target_audience=target_audience,
                num_suggestions=num_suggestions,
                temperature=temperature
            )
    
    def quick_demo(self, business_description: str = None) -> None:
        """Run a quick demo with a sample business description."""
        if business_description is None:
            business_description = "innovative AI-powered restaurant management platform for small businesses"
        
        print(f"🔍 Generating domains for: {business_description}")
        
        # Try to use existing model or create a simple demo
        if self.generator is None:
            print("⚠️  No trained model loaded. This would normally require a trained model.")
            print("📝 Expected output format:")
            sample_domains = [
                {"domain": "restroai.com", "confidence": 0.85},
                {"domain": "kitcheniq.io", "confidence": 0.78},
                {"domain": "smartbites.co", "confidence": 0.72},
                {"domain": "menumaster.app", "confidence": 0.69},
                {"domain": "restotech.com", "confidence": 0.65}
            ]
            
            for i, suggestion in enumerate(sample_domains, 1):
                print(f"  {i}. {suggestion['domain']} (confidence: {suggestion['confidence']:.2f})")
        else:
            suggestions = self.generate_domains(business_description)
            for i, suggestion in enumerate(suggestions, 1):
                if isinstance(suggestion, dict):
                    print(f"  {i}. {suggestion['domain']} (confidence: {suggestion['confidence']:.2f})")
                else:
                    print(f"  {i}. {suggestion}")
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model configuration."""
        return {
            "model_name": self.model_name,
            "base_model": self.config.model.model_name,
            "device": self.config.device,
            "parameters": self._get_model_size(),
            "memory_optimized": "Yes" if self.config.device == "mps" else "No"
        }
    
    def _get_model_size(self) -> str:
        """Get approximate model size information."""
        size_map = {
            "meta-llama/Llama-3.2-1B-Instruct": "1B (~3.5GB)",
            "microsoft/Phi-3-mini-4k-instruct": "3.8B (~3.8GB)"
        }
        return size_map.get(self.config.model.model_name, "Unknown")
    
    def list_available_models(self) -> List[str]:
        """List all available model configurations."""
        return list(self.model_configs.keys())
    
    def benchmark_model(self, test_descriptions: List[str] = None) -> Dict[str, float]:
        """Benchmark model performance on test cases.
        
        Args:
            test_descriptions: List of business descriptions to test
            
        Returns:
            Dictionary with benchmark metrics
        """
        if test_descriptions is None:
            test_descriptions = [
                "modern coffee shop with artisanal pastries",
                "AI-powered financial advisory service",
                "eco-friendly clothing brand for millennials",
                "virtual reality gaming arcade",
                "organic pet food delivery service"
            ]
        
        if self.generator is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        import time
        
        total_time = 0
        total_domains = 0
        
        print("🏁 Running benchmark...")
        
        for i, description in enumerate(test_descriptions, 1):
            start_time = time.time()
            domains = self.generate_domains(description, with_confidence=False)
            end_time = time.time()
            
            generation_time = end_time - start_time
            total_time += generation_time
            total_domains += len(domains)
            
            print(f"  Test {i}/{len(test_descriptions)}: {generation_time:.2f}s, {len(domains)} domains")
        
        avg_time_per_request = total_time / len(test_descriptions)
        avg_domains_per_request = total_domains / len(test_descriptions)
        
        metrics = {
            "avg_time_per_request": avg_time_per_request,
            "avg_domains_per_request": avg_domains_per_request,
            "total_test_time": total_time,
            "domains_per_second": total_domains / total_time if total_time > 0 else 0
        }
        
        print(f"📊 Benchmark Results:")
        print(f"  Average time per request: {metrics['avg_time_per_request']:.2f}s")
        print(f"  Average domains per request: {metrics['avg_domains_per_request']:.1f}")
        print(f"  Domains per second: {metrics['domains_per_second']:.2f}")
        
        return metrics

# Convenience functions for direct Jupyter use
def create_generator(model_name: str = "llama-3.2-1b") -> JupyterDomainGenerator:
    """Create a Jupyter-compatible domain generator.
    
    Args:
        model_name: Model to use ('llama-3.2-1b', 'phi-3-mini', 'dialogpt-medium', 'gpt2-small', 'distilgpt2')
        
    Returns:
        JupyterDomainGenerator instance
    """
    return JupyterDomainGenerator(model_name)

def quick_start_demo() -> None:
    """Run a quick demonstration of the domain generator."""
    print("🚀 Domain Name Generator - Quick Start Demo")
    print("=" * 50)
    
    # Show available models
    generator = JupyterDomainGenerator()
    models = generator.list_available_models()
    print(f"📱 Available models: {', '.join(models)}")
    
    # Show model info
    info = generator.get_model_info()
    print(f"🔧 Current model: {info['base_model']}")
    print(f"💾 Model size: {info['parameters']}")
    print(f"🖥️  Device: {info['device']}")
    print(f"⚡ M1 optimized: {info['memory_optimized']}")
    
    # Run demo
    print("\n🎯 Sample Generation:")
    generator.quick_demo()
    
    print("\n💡 To get started:")
    print("  1. generator = create_generator('llama-3.2-1b')  # or 'phi-3-mini'")
    print("  2. model_path = generator.train_model()         # Train on your data")
    print("  3. generator.load_model(model_path)             # Load trained model") 
    print("  4. domains = generator.generate_domains('your business description')")
    print("\n🔧 Available models: llama-3.2-1b, phi-3-mini")

if __name__ == "__main__":
    quick_start_demo()