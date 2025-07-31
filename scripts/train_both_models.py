#!/usr/bin/env python3
"""
Training script for both Llama and Phi models with W&B and TensorBoard integration
"""

import os
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
import wandb
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from domain_generator.models.trainer import DomainGeneratorTrainer, create_model_configs
from domain_generator.utils.config import Config

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_wandb():
    """Setup Weights & Biases"""
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        logger.warning("WANDB_API_KEY not found. Please set it to use W&B tracking.")
        logger.info("You can get your API key from: https://wandb.ai/authorize")
        return False
    
    project_name = os.getenv("WANDB_PROJECT", "domain-name-generator")
    entity = os.getenv("WANDB_ENTITY")
    
    logger.info(f"W&B Project: {project_name}")
    if entity:
        logger.info(f"W&B Entity: {entity}")
    
    return True

def train_model(model_key: str, config: Config, dataset_path: str, base_output_dir: str):
    """Train a single model with progress tracking"""
    
    model_configs = create_model_configs()
    if model_key not in model_configs:
        raise ValueError(f"Model {model_key} not found in configurations")
    
    model_config = model_configs[model_key]
    
    # Update config with model-specific settings
    config.model.model_name = model_config["model_name"]
    config.lora = model_config["lora_config"]
    config.training = model_config["training_config"]
    
    # Create model-specific output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"{model_key}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 Starting training for {model_key}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"🤖 Model: {model_config['model_name']}")
    
    # Initialize trainer
    trainer = DomainGeneratorTrainer(config)
    
    try:
        # Train the model
        model_path = trainer.train(
            dataset_path=dataset_path,
            output_dir=str(output_dir),
            model_name=model_config["model_name"]
        )
        
        logger.info(f"✅ Training completed for {model_key}")
        logger.info(f"💾 Model saved to: {model_path}")
        
        return {
            "model_key": model_key,
            "model_name": model_config["model_name"],
            "model_path": model_path,
            "output_dir": str(output_dir),
            "status": "completed",
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"❌ Training failed for {model_key}: {str(e)}")
        return {
            "model_key": model_key,
            "model_name": model_config["model_name"],
            "status": "failed",
            "error": str(e),
            "timestamp": timestamp
        }

def generate_training_report(results: list, output_dir: str):
    """Generate a comprehensive training report"""
    
    report = {
        "training_summary": {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(results),
            "successful": len([r for r in results if r["status"] == "completed"]),
            "failed": len([r for r in results if r["status"] == "failed"])
        },
        "model_results": results,
        "system_info": {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "output_directory": output_dir
        }
    }
    
    # Save report
    report_path = Path(output_dir) / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 TRAINING SUMMARY")
    print("="*50)
    print(f"📊 Total models trained: {report['training_summary']['total_models']}")
    print(f"✅ Successful: {report['training_summary']['successful']}")
    print(f"❌ Failed: {report['training_summary']['failed']}")
    print(f"📄 Report saved to: {report_path}")
    print("="*50)
    
    # Model-specific results
    for result in results:
        status_emoji = "✅" if result["status"] == "completed" else "❌"
        print(f"{status_emoji} {result['model_key']}: {result['status']}")
        if result["status"] == "completed":
            print(f"   📁 Path: {result.get('model_path', 'N/A')}")
        else:
            print(f"   ⚠️  Error: {result.get('error', 'Unknown error')}")
    
    print("\n🔗 Access your training progress:")
    print("   • TensorBoard: tensorboard --logdir ./models")
    if os.getenv("WANDB_API_KEY"):
        print("   • Weights & Biases: https://wandb.ai")
    print()
    
    return report_path

def main():
    """Main training pipeline"""
    
    print("🤖 Domain Name Generator - Multi-Model Training")
    print("=" * 50)
    
    # Check W&B setup
    wandb_available = setup_wandb()
    if wandb_available:
        print("✅ Weights & Biases configured")
    else:
        print("⚠️  Weights & Biases not configured (optional)")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "processed" / "training_dataset.json"
    output_dir = project_root / "models"
    
    # Check if dataset exists
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please run data generation first")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Initialize config
    config = Config()
    
    # Models to train (lightweight models due to disk space)
    models_to_train = ["gpt2-small", "distilgpt2"]
    
    print(f"📁 Dataset: {dataset_path}")
    print(f"📂 Output directory: {output_dir}")
    print(f"🎯 Models to train: {', '.join(models_to_train)}")
    print()
    
    # Train each model
    results = []
    for model_key in models_to_train:
        try:
            result = train_model(model_key, config, str(dataset_path), str(output_dir))
            results.append(result)
        except Exception as e:
            logger.error(f"Unexpected error training {model_key}: {e}")
            results.append({
                "model_key": model_key,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            })
    
    # Generate final report
    report_path = generate_training_report(results, str(output_dir))
    
    # Final instructions
    print("🎉 Training pipeline completed!")
    print(f"📊 Full report available at: {report_path}")
    
    # Check if any models completed successfully
    successful_models = [r for r in results if r["status"] == "completed"]
    if successful_models:
        print("\n🚀 Next steps:")
        print("1. View training progress: tensorboard --logdir ./models")
        print("2. Test your models with the inference script")
        print("3. Run evaluation to compare model performance")

if __name__ == "__main__":
    main()