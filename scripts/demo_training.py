#!/usr/bin/env python3
"""
Demonstration training script that shows the training process working
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np

def create_demo_dataset(num_samples: int = 50) -> Dataset:
    """Create a small demo dataset for quick training"""
    
    # Simple domain generation examples
    examples = []
    business_types = [
        "tech startup", "coffee shop", "fitness gym", "online store", 
        "consulting firm", "restaurant", "design agency", "healthcare clinic"
    ]
    
    domain_patterns = [
        "{type}hub.com", "{type}pro.io", "smart{type}.co", "{type}labs.net",
        "the{type}.org", "{type}express.com", "{type}central.io", "mega{type}.co"
    ]
    
    for i in range(num_samples):
        business_type = np.random.choice(business_types)
        pattern = np.random.choice(domain_patterns)
        domain = pattern.format(type=business_type.replace(" ", ""))
        
        prompt = f"Generate a domain name for a {business_type}:"
        completion = f" {domain}"
        
        examples.append({
            "text": f"{prompt}{completion}<|endoftext|>"
        })
    
    return Dataset.from_list(examples)

def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 128) -> Dataset:
    """Tokenize the dataset"""
    
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Remove text column and tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def main():
    print("🤖 Demo Domain Generator Training")
    print("=" * 50)
    
    # Check device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"📱 Device: {device}")
    
    # Load model and tokenizer
    model_name = "distilgpt2"
    print(f"🔧 Loading {model_name}...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Move to device
    model = model.to(device)
    
    print(f"✅ Model loaded: {model.num_parameters():,} parameters")
    
    # Create demo dataset
    print("📚 Creating demo dataset...")
    train_dataset = create_demo_dataset(50)
    eval_dataset = create_demo_dataset(10)
    
    # Tokenize datasets
    print("🔤 Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer)
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Eval samples: {len(eval_dataset)}")
    
    # Setup training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/demo-domain-generator-{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Just 1 epoch for demo
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_steps=10,
        save_steps=25,
        eval_steps=25,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,  # Disable for MPS compatibility
        dataloader_pin_memory=False,
        gradient_checkpointing=False,  # Disable to avoid issues
        report_to=[],  # No reporting
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"📁 Output: {output_dir}")
    
    start_time = time.time()
    
    try:
        trainer.train()
        
        # Save final model
        final_model_path = f"{output_dir}/final"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Training completed!")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"💾 Model saved to: {final_model_path}")
        
        # Quick test generation
        print(f"\n🎯 Testing generation...")
        model.eval()
        
        test_prompt = "Generate a domain name for a tech startup:"
        inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 10,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 Generated: {generated}")
        
        # Save training results
        results = {
            "model_name": model_name,
            "model_path": final_model_path,
            "output_dir": output_dir,
            "training_duration": duration,
            "device": device,
            "success": True,
            "timestamp": timestamp,
            "test_generation": generated
        }
        
        results_dir = Path("data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"demo_training_result_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        return final_model_path
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ Training failed: {str(e)}")
        
        # Save error results
        results = {
            "model_name": model_name,
            "model_path": None,
            "output_dir": output_dir,
            "training_duration": duration,
            "device": device,
            "success": False,
            "error": str(e),
            "timestamp": timestamp
        }
        
        results_dir = Path("data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"demo_training_result_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Error results saved to: {results_file}")
        return None

if __name__ == "__main__":
    main()