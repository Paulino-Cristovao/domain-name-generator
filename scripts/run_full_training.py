#!/usr/bin/env python3
"""
Full training pipeline for both Llama-3.2-1B and Phi-3-Mini models
with comprehensive W&B tracking and model checkpointing.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import wandb
from dotenv import load_dotenv

from domain_generator.models.trainer import DomainGeneratorTrainer, create_model_configs
from domain_generator.utils.config import Config

# Load environment variables
load_dotenv()

def setup_wandb() -> None:
    """Initialize W&B with project configuration"""
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY not found in environment variables")
    
    print(f"🔑 W&B API Key: {wandb_api_key[:8]}...")
    print(f"📊 W&B Project: {os.getenv('WANDB_PROJECT', 'domain-name-generator')}")

def train_model(model_key: str, model_config: Dict[str, Any], base_config: Config) -> Dict[str, Any]:
    """Train a single model with W&B tracking"""
    
    print(f"\n🚀 Starting training for {model_key}")
    print("=" * 50)
    
    # Update config with model-specific settings
    base_config.model.model_name = model_config["model_name"]
    base_config.lora = model_config["lora_config"]
    base_config.training = model_config["training_config"]
    
    # Create trainer
    trainer = DomainGeneratorTrainer(base_config)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/{model_key}-domain-generator-{timestamp}"
    
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 Model: {model_config['model_name']}")
    print(f"⚙️  Device: {base_config.device}")
    print(f"🎯 Epochs: {base_config.training.num_epochs}")
    print(f"📚 Batch size: {base_config.training.per_device_train_batch_size}")
    print(f"🔄 Gradient accumulation: {base_config.training.gradient_accumulation_steps}")
    print(f"📈 Learning rate: {base_config.training.learning_rate}")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Train the model
        model_path = trainer.train(
            dataset_path="data/processed/training_dataset.json",
            output_dir=output_dir,
            model_name=model_config["model_name"]
        )
        
        # Record end time
        end_time = time.time()
        training_duration = end_time - start_time
        
        # Create training results
        results = {
            "model_key": model_key,
            "model_name": model_config["model_name"],
            "model_path": model_path,
            "output_dir": output_dir,
            "training_duration": training_duration,
            "training_duration_formatted": f"{training_duration // 3600:.0f}h {(training_duration % 3600) // 60:.0f}m {training_duration % 60:.0f}s",
            "epochs": base_config.training.num_epochs,
            "device": base_config.device,
            "success": True,
            "timestamp": timestamp,
            "wandb_run_url": wandb.run.url if wandb.run else None
        }
        
        print(f"✅ Training completed successfully!")
        print(f"⏱️  Duration: {results['training_duration_formatted']}")
        print(f"💾 Model saved to: {model_path}")
        if results['wandb_run_url']:
            print(f"📊 W&B Run: {results['wandb_run_url']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        
        # Record failure
        end_time = time.time()
        training_duration = end_time - start_time
        
        results = {
            "model_key": model_key,
            "model_name": model_config["model_name"],
            "model_path": None,
            "output_dir": output_dir,
            "training_duration": training_duration,
            "training_duration_formatted": f"{training_duration // 3600:.0f}h {(training_duration % 3600) // 60:.0f}m {training_duration % 60:.0f}s",
            "epochs": base_config.training.num_epochs,
            "device": base_config.device,
            "success": False,
            "error": str(e),
            "timestamp": timestamp,
            "wandb_run_url": wandb.run.url if wandb.run else None
        }
        
        return results

def save_training_results(results: Dict[str, Any]) -> str:
    """Save training results to file"""
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"training_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Training results saved to: {results_file}")
    return str(results_file)

def generate_comparative_report(results: Dict[str, Any]) -> str:
    """Generate comparative training report"""
    
    print("\n📊 Generating Comparative Training Report")
    print("=" * 50)
    
    report_content = []
    report_content.append("# Domain Name Generator: Training Results Report")
    report_content.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"**W&B Project**: {os.getenv('WANDB_PROJECT', 'domain-name-generator')}")
    
    # Executive Summary
    report_content.append("\n## Executive Summary")
    
    successful_models = [k for k, v in results.items() if v.get('success', False)]
    failed_models = [k for k, v in results.items() if not v.get('success', False)]
    
    report_content.append(f"\n- **Total Models Trained**: {len(results)}")
    report_content.append(f"- **Successful**: {len(successful_models)} ({', '.join(successful_models)})")
    if failed_models:
        report_content.append(f"- **Failed**: {len(failed_models)} ({', '.join(failed_models)})")
    
    # Training Configuration
    report_content.append("\n## Training Configuration")
    report_content.append("\n| Model | Parameters | Epochs | Batch Size | Learning Rate | Duration |")
    report_content.append("|-------|------------|--------|------------|---------------|----------|")
    
    for model_key, result in results.items():
        if result.get('success'):
            # Get model size info
            if 'llama' in model_key.lower():
                params = "1B"
            elif 'phi' in model_key.lower():
                params = "3.8B"
            else:
                params = "Unknown"
            
            # Get training config from model configs
            model_configs = create_model_configs()
            if model_key in model_configs:
                config = model_configs[model_key]
                batch_size = config["training_config"].per_device_train_batch_size
                learning_rate = config["training_config"].learning_rate
            else:
                batch_size = "Unknown"
                learning_rate = "Unknown"
            
            report_content.append(f"| **{model_key}** | {params} | {result['epochs']} | {batch_size} | {learning_rate} | {result['training_duration_formatted']} |")
    
    # W&B Links
    report_content.append("\n## W&B Experiment Links")
    for model_key, result in results.items():
        if result.get('wandb_run_url'):
            report_content.append(f"- **{model_key}**: [{result['wandb_run_url']}]({result['wandb_run_url']})")
    
    # Model Checkpoints
    report_content.append("\n## Model Checkpoints")
    for model_key, result in results.items():
        if result.get('success') and result.get('model_path'):
            report_content.append(f"- **{model_key}**: `{result['model_path']}`")
    
    # Performance Analysis
    report_content.append("\n## Performance Analysis")
    
    if len(successful_models) >= 2:
        # Compare training times
        training_times = {k: v['training_duration'] for k, v in results.items() if v.get('success')}
        fastest_model = min(training_times.keys(), key=lambda k: training_times[k])
        slowest_model = max(training_times.keys(), key=lambda k: training_times[k])
        
        report_content.append(f"\n### Training Speed Comparison")
        report_content.append(f"- **Fastest**: {fastest_model} ({results[fastest_model]['training_duration_formatted']})")
        report_content.append(f"- **Slowest**: {slowest_model} ({results[slowest_model]['training_duration_formatted']})")
        
        # Speed difference
        speed_ratio = training_times[slowest_model] / training_times[fastest_model]
        report_content.append(f"- **Speed Difference**: {speed_ratio:.1f}x slower")
    
    # Hardware Information
    base_config = Config()
    report_content.append(f"\n### Hardware Configuration")
    report_content.append(f"- **Device**: {base_config.device}")
    report_content.append(f"- **Optimization**: M1 MPS acceleration with bfloat16 precision")
    report_content.append(f"- **Memory**: <8GB (M1 optimized)")
    
    # Next Steps
    report_content.append("\n## Next Steps")
    report_content.append("\n1. **Model Evaluation**: Run GPT-4o LLM-as-a-Judge evaluation on both models")
    report_content.append("2. **Performance Testing**: Benchmark inference speed and quality")
    report_content.append("3. **Edge Case Analysis**: Test both models on comprehensive edge cases")
    report_content.append("4. **Production Deployment**: Deploy best-performing model")
    
    # Errors (if any)
    if failed_models:
        report_content.append("\n## Training Errors")
        for model_key in failed_models:
            result = results[model_key]
            report_content.append(f"\n### {model_key}")
            report_content.append(f"- **Error**: {result.get('error', 'Unknown error')}")
            report_content.append(f"- **Duration before failure**: {result['training_duration_formatted']}")
    
    # Save report
    report_text = "\n".join(report_content)
    
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f"training_comparative_report_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"📝 Comparative report saved to: {report_file}")
    return str(report_file)

def main():
    """Main training pipeline"""
    
    print("🤖 Domain Name Generator: Full Training Pipeline")
    print("=" * 60)
    
    # Setup W&B
    setup_wandb()
    
    # Initialize base config
    base_config = Config()
    
    # Get model configurations
    model_configs = create_model_configs()
    
    print(f"\n📱 Device: {base_config.device}")
    print(f"🔧 Models to train: {list(model_configs.keys())}")
    
    # Train all models
    results = {}
    
    for model_key, model_config in model_configs.items():
        print(f"\n{'='*20} {model_key.upper()} {'='*20}")
        
        # Update todo
        print(f"🎯 Starting {model_key} training...")
        
        result = train_model(model_key, model_config, base_config)
        results[model_key] = result
        
        # Small break between models
        if len(model_configs) > 1:
            print("\n⏸️  Brief pause before next model...")
            time.sleep(5)
    
    # Save results
    results_file = save_training_results(results)
    
    # Generate comparative report
    report_file = generate_comparative_report(results)
    
    # Final summary
    print(f"\n🎉 Training Pipeline Complete!")
    print("=" * 60)
    print(f"📊 Results file: {results_file}")
    print(f"📝 Comparative report: {report_file}")
    
    successful_models = [k for k, v in results.items() if v.get('success', False)]
    if successful_models:
        print(f"✅ Successfully trained: {', '.join(successful_models)}")
        
        print(f"\n🔗 W&B Links:")
        for model_key, result in results.items():
            if result.get('wandb_run_url'):
                print(f"  {model_key}: {result['wandb_run_url']}")
    
    failed_models = [k for k, v in results.items() if not v.get('success', False)]
    if failed_models:
        print(f"❌ Failed models: {', '.join(failed_models)}")

if __name__ == "__main__":
    main()