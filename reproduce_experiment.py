#!/usr/bin/env python3
# Auto-generated reproduction script

import sys
import torch
import numpy as np
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Add src to path
sys.path.append('src')

from domain_generator.models.jupyter_compatible import create_generator

def reproduce_experiment(model_config_id, dataset_path="data/processed/training_dataset.json"):
    """Reproduce a training experiment""" 
    print(f"Reproducing experiment for {model_config_id}")
    
    generator = create_generator(model_config_id)
    model_path = generator.train_model(
        dataset_path=dataset_path,
        output_dir=f"models/reproduced_{model_config_id}",
        use_wandb=False  # Disable W&B for reproduction
    )
    
    print(f"Model saved to: {model_path}")
    return model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reproduce training experiment")
    parser.add_argument("model_config", help="Model configuration ID")
    parser.add_argument("--dataset", default="data/processed/training_dataset.json", 
                       help="Path to training dataset")
    
    args = parser.parse_args()
    reproduce_experiment(args.model_config, args.dataset)