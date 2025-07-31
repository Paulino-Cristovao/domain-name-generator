#!/usr/bin/env python3
"""
Single model training script with proper W&B configuration
"""

import os
import sys
import json
import time
import argparse
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

def main():
    parser = argparse.ArgumentParser(description='Train a single domain generation model')
    parser.add_argument('model_key', choices=['gpt2-medium', 'distilgpt2'], 
                       help='Model to train')
    parser.add_argument('--no-wandb', action='store_true', 
                       help='Disable W&B logging')
    
    args = parser.parse_args()
    
    print(f"🤖 Training {args.model_key}")
    print("=" * 50)
    
    # Initialize base config
    base_config = Config()
    
    # Get model configuration
    model_configs = create_model_configs()
    if args.model_key not in model_configs:
        print(f"❌ Model {args.model_key} not found")
        return
    
    model_config = model_configs[args.model_key]
    
    # Update config with model-specific settings
    base_config.model.model_name = model_config["model_name"]
    base_config.lora = model_config["lora_config"]
    base_config.training = model_config["training_config"]
    
    print(f"🔧 Model: {model_config['model_name']}")
    print(f"⚙️  Device: {base_config.device}")
    print(f"🎯 Epochs: {base_config.training.num_epochs}")
    print(f"📚 Batch size: {base_config.training.per_device_train_batch_size}")
    print(f"🔄 Gradient accumulation: {base_config.training.gradient_accumulation_steps}")
    print(f"📈 Learning rate: {base_config.training.learning_rate}")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/{args.model_key}-domain-generator-{timestamp}"
    
    print(f"📁 Output directory: {output_dir}")
    
    if not args.no_wandb:
        # Check W&B configuration
        wandb_api_key = os.getenv("WANDB_API_KEY")
        wandb_entity = os.getenv("WANDB_ENTITY")
        
        if not wandb_api_key:
            print("⚠️  WANDB_API_KEY not found, disabling W&B")
            args.no_wandb = True
        else:
            print(f"🔑 W&B API Key: {wandb_api_key[:8]}...")
            print(f"👤 W&B Entity: {wandb_entity}")
            print(f"📊 W&B Project: {os.getenv('WANDB_PROJECT', 'domain-name-generator')}")
    
    # Initialize trainer
    trainer = DomainGeneratorTrainer(base_config)
    
    # Record start time
    start_time = time.time()
    
    try:
        print(f"\n🚀 Starting training...")
        
        # Train the model
        model_path = trainer.train(
            dataset_path="data/processed/training_dataset.json",
            output_dir=output_dir,
            model_name=model_config["model_name"]
        )
        
        # Record end time
        end_time = time.time()
        training_duration = end_time - start_time
        
        print(f"✅ Training completed successfully!")
        print(f"⏱️  Duration: {training_duration // 3600:.0f}h {(training_duration % 3600) // 60:.0f}m {training_duration % 60:.0f}s")
        print(f"💾 Model saved to: {model_path}")
        
        # Save training results
        results = {
            "model_key": args.model_key,
            "model_name": model_config["model_name"],
            "model_path": model_path,
            "output_dir": output_dir,
            "training_duration": training_duration,
            "training_duration_formatted": f"{training_duration // 3600:.0f}h {(training_duration % 3600) // 60:.0f}m {training_duration % 60:.0f}s",
            "epochs": base_config.training.num_epochs,
            "device": base_config.device,
            "success": True,
            "timestamp": timestamp,
            "wandb_disabled": args.no_wandb
        }
        
        # Save results to file
        results_dir = Path("data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"{args.model_key}_training_result_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        
        # Record failure
        end_time = time.time()
        training_duration = end_time - start_time
        
        results = {
            "model_key": args.model_key,
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
            "wandb_disabled": args.no_wandb
        }
        
        # Save error results to file
        results_dir = Path("data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"{args.model_key}_training_result_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Error results saved to: {results_file}")
        return None

if __name__ == "__main__":
    main()