#!/usr/bin/env python3
"""
Simple training script for Phi-2 model - 1 epoch
Workaround for numpy compatibility issues
"""

import os
import json
import time
import torch
from typing import List, Dict

def create_phi2_training_script():
    """Create a standalone training script that avoids the numpy issue"""
    
    print("🧠 Phi-2 Training Setup (1 epoch)")
    print("=" * 50)
    
    # Check if we can import transformers at all
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ Transformers library accessible")
    except Exception as e:
        print(f"❌ Cannot import transformers: {e}")
        print("💡 Consider running in a different environment (like Google Colab)")
        return False
    
    # Create a simple training configuration
    config = {
        "model_name": "microsoft/phi-2",
        "dataset_path": "data/processed/phi2_mistral_training_dataset.json",
        "output_dir": "models/phi-2-domain-generator",
        "max_length": 512,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 2e-4
    }
    
    print(f"📊 Model: {config['model_name']}")
    print(f"📁 Dataset: {config['dataset_path']}")
    print(f"💾 Output: {config['output_dir']}")
    print(f"⚡ Epochs: {config['num_epochs']}")
    
    # Check dataset exists
    if not os.path.exists(config["dataset_path"]):
        print(f"❌ Dataset not found: {config['dataset_path']}")
        return False
    
    # Load dataset
    with open(config["dataset_path"], 'r') as f:
        data = json.load(f)
    
    print(f"📈 Dataset loaded: {len(data)} examples")
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Save training configuration
    config_path = os.path.join(config["output_dir"], "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️  Configuration saved: {config_path}")
    
    # For now, simulate training completion since we have environment issues
    print("\n🔄 Training would start here...")
    print("⚠️  Note: Due to numpy compatibility issues in this environment,")
    print("   actual model training should be run in Google Colab or a fresh environment.")
    
    # Create mock model directory structure
    final_dir = os.path.join(config["output_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)
    
    # Create a training summary
    training_summary = {
        "model_name": config["model_name"],
        "training_examples": len(data),
        "epochs": config["num_epochs"],
        "status": "environment_limited",
        "note": "Training configured but needs clean environment to execute",
        "recommended_platform": "Google Colab",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = os.path.join(final_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"📋 Training summary saved: {summary_path}")
    
    return True

if __name__ == "__main__":
    success = create_phi2_training_script()
    if success:
        print("\n✅ Phi-2 training setup completed")
        print("💡 For actual training, use Google Colab notebook: domain_generator_colab_open_access.ipynb")
    else:
        print("\n❌ Phi-2 training setup failed")