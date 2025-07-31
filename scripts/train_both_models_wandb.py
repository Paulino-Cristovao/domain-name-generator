#!/usr/bin/env python3
"""
Training script for both Llama and Phi models with W&B tracking and TensorBoard visualization
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from domain_generator.models.trainer import DomainGeneratorTrainer, create_model_configs
from domain_generator.utils.config import Config

def setup_wandb():
    """Setup Weights & Biases with environment variables"""
    import wandb
    
    # Check if W&B API key is set
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("⚠️  WANDB_API_KEY not found in environment variables")
        print("To set up W&B tracking, run:")
        print("export WANDB_API_KEY='your_api_key_here'")
        print("export WANDB_PROJECT='domain-name-generator'")
        print("export WANDB_ENTITY='your_username_or_team'")
        return False
    
    # Set default project if not specified
    if not os.getenv("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = "domain-name-generator"
    
    print(f"✅ W&B configured:")
    print(f"   Project: {os.getenv('WANDB_PROJECT')}")
    print(f"   Entity: {os.getenv('WANDB_ENTITY', 'default')}")
    
    return True

def train_single_model(model_key: str, config: Config, dataset_path: str, models_dir: str):
    """Train a single model with W&B tracking"""
    print(f"\n{'='*60}")
    print(f"Training {model_key.upper()} Model")
    print(f"{'='*60}")
    
    model_configs = create_model_configs()
    if model_key not in model_configs:
        raise ValueError(f"Model {model_key} not found in configurations")
    
    model_config = model_configs[model_key]
    
    # Update config with selected model settings
    config.model.model_name = model_config["model_name"]
    config.lora = model_config["lora_config"]
    config.training = model_config["training_config"]
    
    # Create model-specific output directory
    output_dir = f"{models_dir}/{model_key}-domain-generator"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = DomainGeneratorTrainer(config)
    
    # Train the model
    start_time = time.time()
    model_path = trainer.train(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_name=model_config["model_name"]
    )
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"✅ {model_key} training completed in {training_time/60:.1f} minutes")
    print(f"📁 Model saved to: {model_path}")
    
    # Save training summary
    summary = {
        "model_key": model_key,
        "model_name": model_config["model_name"],
        "model_path": model_path,
        "training_time_minutes": training_time / 60,
        "training_completed": datetime.now().isoformat(),
        "tensorboard_logs": f"{output_dir}/logs",
        "config": {
            "lora_r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.per_device_train_batch_size,
            "epochs": config.training.num_epochs
        }
    }
    
    with open(f"{output_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary

def generate_comparative_report(summaries: list, output_dir: str):
    """Generate a comparative report between models"""
    print(f"\n{'='*60}")
    print("Generating Comparative Report")
    print(f"{'='*60}")
    
    report = {
        "training_session": datetime.now().isoformat(),
        "models_trained": len(summaries),
        "model_comparisons": [],
        "summary": {}
    }
    
    total_time = 0
    for summary in summaries:
        total_time += summary["training_time_minutes"]
        
        model_info = {
            "model_key": summary["model_key"],
            "model_name": summary["model_name"],
            "training_time_minutes": summary["training_time_minutes"],
            "model_path": summary["model_path"],
            "tensorboard_logs": summary["tensorboard_logs"],
            "configuration": summary["config"]
        }
        report["model_comparisons"].append(model_info)
    
    # Add summary statistics
    report["summary"] = {
        "total_training_time_minutes": total_time,
        "average_training_time_minutes": total_time / len(summaries),
        "fastest_model": min(summaries, key=lambda x: x["training_time_minutes"])["model_key"],
        "slowest_model": max(summaries, key=lambda x: x["training_time_minutes"])["model_key"]
    }
    
    # Save report
    report_path = f"{output_dir}/comparative_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"📊 Training Summary:")
    print(f"   Total models trained: {len(summaries)}")
    print(f"   Total training time: {total_time:.1f} minutes")
    print(f"   Average time per model: {total_time/len(summaries):.1f} minutes")
    print(f"   Fastest model: {report['summary']['fastest_model']}")
    print(f"   Slowest model: {report['summary']['slowest_model']}")
    print(f"📁 Report saved to: {report_path}")
    
    return report_path

def print_visualization_links(models_dir: str):
    """Print links to visualization tools"""
    print(f"\n{'='*60}")
    print("Visualization Links")
    print(f"{'='*60}")
    
    print("🔗 TensorBoard Visualization:")
    for model_key in ["llama-1b", "phi-1.5"]:
        log_dir = f"{models_dir}/{model_key}-domain-generator/logs"
        if os.path.exists(log_dir):
            print(f"   {model_key}: tensorboard --logdir {log_dir}")
    
    print(f"\n📊 To start TensorBoard for all models:")
    print(f"   tensorboard --logdir {models_dir}")
    print(f"   Then open: http://localhost:6006")
    
    wandb_project = os.getenv("WANDB_PROJECT", "domain-name-generator")
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    
    if wandb_entity:
        print(f"\n🚀 Weights & Biases Dashboard:")
        print(f"   https://wandb.ai/{wandb_entity}/{wandb_project}")
    else:
        print(f"\n🚀 Weights & Biases Dashboard:")
        print(f"   https://wandb.ai/[your-username]/{wandb_project}")

def main():
    """Main training function"""
    print("🚀 Starting Multi-Model Training with W&B and TensorBoard")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Setup directories
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "processed" / "training_dataset.json"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Check dataset exists
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure the dataset is generated first")
        return
    
    # Setup W&B
    wandb_available = setup_wandb()
    if not wandb_available:
        print("⚠️  Continuing without W&B tracking")
    
    # Initialize config
    config = Config()
    
    # Models to train
    models_to_train = ["llama-1b", "phi-1.5"]
    training_summaries = []
    
    # Train each model
    for model_key in models_to_train:
        try:
            summary = train_single_model(
                model_key=model_key,
                config=config,
                dataset_path=str(dataset_path),
                models_dir=str(models_dir)
            )
            training_summaries.append(summary)
        except Exception as e:
            print(f"❌ Error training {model_key}: {e}")
            continue
    
    # Generate comparative report
    if training_summaries:
        report_path = generate_comparative_report(training_summaries, str(models_dir))
        
        # Print visualization links
        print_visualization_links(str(models_dir))
        
        print(f"\n✅ All training completed successfully!")
        print(f"📊 Comparative report: {report_path}")
    else:
        print("❌ No models were trained successfully")

if __name__ == "__main__":
    main()