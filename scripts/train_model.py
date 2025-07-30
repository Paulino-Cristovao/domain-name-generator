#!/usr/bin/env python3
"""Training script for domain generation models"""
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from domain_generator.models.trainer import DomainGeneratorTrainer, create_model_configs
from domain_generator.utils.config import Config
import json

def main():
    parser = argparse.ArgumentParser(description='Train domain generation model')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['mistral-7b', 'llama-3.1-8b'],
                       help='Model to train')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to training dataset JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Validate dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Load model configuration
    model_configs = create_model_configs()
    if args.model not in model_configs:
        print(f"Error: Unknown model: {args.model}")
        sys.exit(1)
    
    model_config = model_configs[args.model]
    
    # Initialize config
    config = Config()
    config.model.model_name = model_config["model_name"]
    config.lora = model_config["lora_config"]
    config.training = model_config["training_config"]
    
    # Override config with command line arguments
    config.training.num_epochs = args.epochs
    config.training.per_device_train_batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"models/{args.model}-domain-generator"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Training {args.model} model...")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.training.per_device_train_batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Initialize trainer
    trainer = DomainGeneratorTrainer(config)
    
    try:
        # Train the model
        model_path = trainer.train(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            model_name=model_config["model_name"]
        )
        
        # Save training configuration
        config_path = os.path.join(args.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            config_dict = {
                "model_name": config.model.model_name,
                "dataset_path": args.dataset,
                "training_args": {
                    "num_epochs": config.training.num_epochs,
                    "batch_size": config.training.per_device_train_batch_size,
                    "learning_rate": config.training.learning_rate,
                    "gradient_accumulation_steps": config.training.gradient_accumulation_steps
                },
                "lora_config": {
                    "r": config.lora.r,
                    "lora_alpha": config.lora.lora_alpha,
                    "target_modules": config.lora.target_modules,
                    "lora_dropout": config.lora.lora_dropout
                }
            }
            json.dump(config_dict, f, indent=2)
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Configuration saved to: {config_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()