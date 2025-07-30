"""Model training with LoRA fine-tuning optimized for Mac M1"""
import os
import json
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import numpy as np
from pathlib import Path
import wandb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ..utils.config import Config, ModelConfig, LoRAConfig, TrainingConfig

class DomainGeneratorTrainer:
    """Fine-tune LLMs for domain name generation using LoRA"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.device = config.device
        
    def setup_model_and_tokenizer(self, model_name: str):
        """Initialize model and tokenizer with memory optimizations"""
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with memory optimizations for Mac M1
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,  # Memory optimization
            "device_map": "auto" if self.device != "mps" else None,
            "low_cpu_mem_usage": True
        }
        
        # Adjust for specific models
        if "phi" in model_name.lower():
            model_kwargs["attn_implementation"] = "eager"  # Phi-3 compatibility
            model_kwargs["torch_dtype"] = torch.bfloat16  # Phi-3 prefers bfloat16
        elif "llama" in model_name.lower():
            model_kwargs["torch_dtype"] = torch.bfloat16  # Llama prefers bfloat16
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to MPS if available (Mac M1)
        if self.device == "mps":
            self.model = self.model.to(self.device)
            
        print(f"Model loaded on device: {self.device}")
        print(f"Model parameters: {self.model.num_parameters():,}")
        
    def setup_lora(self, lora_config: LoRAConfig):
        """Setup LoRA configuration for memory-efficient fine-tuning"""
        
        # Adjust target modules based on model architecture
        target_modules = lora_config.target_modules.copy()
        
        # Model-specific target modules
        model_name_lower = self.config.model.model_name.lower()
        
        if "phi" in model_name_lower:
            # Phi-3 uses different attention module names
            target_modules = ["qkv_proj", "o_proj"]
        elif "llama" in model_name_lower:
            # Llama uses standard transformer attention modules
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Check actual layer structure for other models
            first_layer = self.model.model.layers[0]
            available_modules = [name for name, _ in first_layer.named_modules()]
            
            # Filter target modules to only include available ones
            target_modules = [mod for mod in target_modules if any(mod in avail for avail in available_modules)]
            
            if not target_modules:
                # Fallback to common attention modules
                target_modules = ["q_proj", "v_proj"]
                
        print(f"Using target modules: {target_modules}")
        
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()
        
    def prepare_dataset(self, dataset_path: str) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        print(f"Loading dataset from: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Create training examples
        train_texts = []
        for example in data:
            # Format as instruction-following conversation
            text = f"{example['prompt']}\n{example['completion']}"
            train_texts.append(text)
        
        # Split into train/val (90/10)
        split_idx = int(len(train_texts) * 0.9)
        train_texts = train_texts[:split_idx]
        val_texts = train_texts[split_idx:]
        
        # Tokenize datasets
        train_dataset = self._tokenize_texts(train_texts)
        val_dataset = self._tokenize_texts(val_texts)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def _tokenize_texts(self, texts: List[str]) -> Dataset:
        """Tokenize text data for training"""
        def tokenize_function(examples):
            # Tokenize with padding and truncation
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.model.max_length,
                return_tensors="pt"
            )
            
            # Set labels for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return dataset
    
    def create_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Create training arguments optimized for Mac M1"""
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps,
            logging_steps=self.config.training.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            fp16=self.config.training.fp16 and self.device != "mps",  # FP16 not supported on MPS
            bf16=self.device == "mps",  # Use BF16 on Mac M1
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            report_to=["wandb", "tensorboard"],
            logging_dir=f"{output_dir}/logs",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )
    
    def train(self, dataset_path: str, output_dir: str, model_name: str) -> str:
        """Complete training pipeline with W&B integration"""
        
        # Initialize Weights & Biases
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "domain-name-generator"),
            entity=os.getenv("WANDB_ENTITY"),
            config={
                "model_name": model_name,
                "dataset_path": dataset_path,
                "lora_r": self.config.lora.r,
                "lora_alpha": self.config.lora.lora_alpha,
                "learning_rate": self.config.training.learning_rate,
                "batch_size": self.config.training.per_device_train_batch_size,
                "epochs": self.config.training.num_epochs,
                "device": self.device
            }
        )
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer(model_name)
        
        # Setup LoRA
        self.setup_lora(self.config.lora)
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_dataset(dataset_path)
        
        # Create training arguments
        training_args = self.create_training_arguments(output_dir)
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Create trainer with checkpoint callback
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                CheckpointSaverCallback(save_top_k=2)  # Save top 2 checkpoints
            ]
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save final model
        final_model_path = f"{output_dir}/final"
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        # Log final metrics to W&B
        wandb.log({
            "final_model_path": final_model_path,
            "training_completed": True
        })
        
        # Finish W&B run
        wandb.finish()
        
        print(f"Training completed. Model saved to: {final_model_path}")
        return final_model_path
    
    def load_trained_model(self, model_path: str, base_model_name: str):
        """Load a trained LoRA model for inference"""
        print(f"Loading trained model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device != "mps" else None,
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        if self.device == "mps":
            self.model = self.model.to(self.device)
        
        print("Model loaded successfully")

class CheckpointSaverCallback:
    """Custom callback to save top-k checkpoints based on evaluation metrics"""
    
    def __init__(self, save_top_k: int = 2):
        self.save_top_k = save_top_k
        self.best_scores = []
        
    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        """Save checkpoint if it's among the top-k performers"""
        current_score = logs.get("eval_loss", float('inf'))
        
        # Keep track of best scores
        self.best_scores.append((state.global_step, current_score))
        self.best_scores.sort(key=lambda x: x[1])  # Sort by score (lower is better for loss)
        
        # Keep only top-k
        if len(self.best_scores) > self.save_top_k:
            self.best_scores = self.best_scores[:self.save_top_k]
        
        # Check if current checkpoint should be saved
        if (state.global_step, current_score) in self.best_scores:
            print(f"Saving top-{self.save_top_k} checkpoint at step {state.global_step} with score {current_score:.4f}")
            
            # Save additional info about this checkpoint
            checkpoint_info = {
                "step": state.global_step,
                "eval_loss": current_score,
                "rank_in_top_k": self.best_scores.index((state.global_step, current_score)) + 1
            }
            
            checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"
            os.makedirs(checkpoint_path, exist_ok=True)
            
            with open(f"{checkpoint_path}/checkpoint_info.json", "w") as f:
                json.dump(checkpoint_info, f, indent=2)

def create_model_configs() -> Dict[str, Dict]:
    """Create configurations for different models optimized for Mac M1 8GB with <4GB models"""
    
    configs = {
        "dialogpt-medium": {
            "model_name": "microsoft/DialoGPT-medium",  # 355M parameters (~1.4GB)
            "lora_config": LoRAConfig(
                r=16,
                lora_alpha=32,
                target_modules=["c_attn", "c_proj"],  # GPT-2 style attention modules
                lora_dropout=0.1
            ),
            "training_config": TrainingConfig(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                learning_rate=3e-4,
                num_epochs=2
            )
        },
        "gpt2-small": {
            "model_name": "gpt2",  # 124M parameters (~500MB)
            "lora_config": LoRAConfig(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.1
            ),
            "training_config": TrainingConfig(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
                learning_rate=5e-4,
                num_epochs=2
            )
        },
        "distilgpt2": {
            "model_name": "distilgpt2",  # 82M parameters (~330MB)
            "lora_config": LoRAConfig(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.05
            ),
            "training_config": TrainingConfig(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
                learning_rate=5e-4,
                num_epochs=2
            )
        },
        "llama-3.2-1b": {
            "model_name": "meta-llama/Llama-3.2-1B-Instruct",  # 1B parameters (~3.5GB)
            "lora_config": LoRAConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Llama attention modules
                lora_dropout=0.1
            ),
            "training_config": TrainingConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                num_epochs=2
            )
        },
        "phi-3-mini": {
            "model_name": "microsoft/Phi-3-mini-4k-instruct",  # 3.8B parameters (~3.8GB)
            "lora_config": LoRAConfig(
                r=16,
                lora_alpha=32,
                target_modules=["qkv_proj", "o_proj"],  # Phi-3 attention modules
                lora_dropout=0.1
            ),
            "training_config": TrainingConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                learning_rate=1e-4,
                num_epochs=2
            )
        }
    }
    
    return configs

if __name__ == "__main__":
    # Example usage
    config = Config()
    trainer = DomainGeneratorTrainer(config)
    
    # Train DialoGPT-medium model (M1 optimized)
    model_configs = create_model_configs()
    dialogpt_config = model_configs["dialogpt-medium"]
    
    # Update config with DialoGPT settings
    config.model.model_name = dialogpt_config["model_name"]
    config.lora = dialogpt_config["lora_config"]
    config.training = dialogpt_config["training_config"]
    
    # Train the model
    model_path = trainer.train(
        dataset_path="data/processed/training_dataset.json",
        output_dir="models/dialogpt-medium-domain-generator",
        model_name=dialogpt_config["model_name"]
    )
    
    print(f"DialoGPT-medium model training completed: {model_path}")