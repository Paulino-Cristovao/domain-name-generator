#!/usr/bin/env python3
"""
Run Phi-2 training for 1 epoch with progress bars
"""

import os
import sys
import json
import time
import torch
from typing import List, Dict
from tqdm.auto import tqdm

# Add src to path
sys.path.append('src')

def run_phi2_training():
    """Run Phi-2 training with the current environment"""
    
    print("🧠 Starting Phi-2 Training (1 epoch)")
    print("=" * 50)
    
    try:
        from transformers import (
            AutoTokenizer, 
            AutoModelForCausalLM, 
            TrainingArguments, 
            Trainer,
            DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        
        print("✅ All imports successful")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Configuration
    config = {
        "model_name": "microsoft/phi-2",
        "dataset_path": "data/processed/phi2_mistral_training_dataset.json",
        "output_dir": "models/phi-2-domain-generator-trained",
        "max_length": 512,
        "num_epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "save_steps": 50,  # More frequent saves for 1 epoch
        "logging_steps": 10
    }
    
    print(f"📊 Model: {config['model_name']}")
    print(f"⚡ Epochs: {config['num_epochs']}")
    print(f"📁 Output: {config['output_dir']}")
    
    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        print("🖥️  Using MPS (Mac M1/M2)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🖥️  Using CUDA")
    else:
        device = "cpu"
        print("🖥️  Using CPU")
    
    # Load dataset
    print("\n📊 Loading dataset...")
    with open(config["dataset_path"], 'r') as f:
        data = json.load(f)
    
    print(f"📈 Dataset size: {len(data)} examples")
    
    # Prepare texts
    texts = []
    for item in tqdm(data, desc="Processing dataset"):
        text = f"{item['prompt']}\n\n{item['completion']}"
        texts.append(text)
    
    print(f"✅ Processed {len(texts)} training examples")
    
    try:
        # Load tokenizer
        print("\n🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Tokenizer loaded")
        
        # Load model
        print("🧠 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if device != "mps" else None
        )
        
        if device == "mps":
            model = model.to("mps")
        
        print(f"✅ Model loaded on {device}")
        print(f"📊 Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # Setup LoRA
        print("\n🎯 Setting up LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["Wqkv", "out_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ LoRA setup complete")
        print(f"🎯 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Tokenize dataset
        print("\n🔄 Tokenizing dataset...")
        def tokenize_function(examples):
            result = tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=config["max_length"]
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=10,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        print(f"✅ Dataset tokenized: {len(tokenized_dataset)} examples")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=config["logging_steps"],
            save_steps=config["save_steps"],
            save_total_limit=2,
            fp16=False,  # Disable FP16 for stability
            bf16=device == "mps",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
            disable_tqdm=False,  # Enable tqdm progress bars
            dataloader_num_workers=0
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create trainer
        print("\n🏋️ Setting up trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        print("✅ Trainer ready")
        
        # Start training
        print(f"\n🚀 Starting training...")
        print(f"⏱️  Estimated time: ~15-20 minutes on M1/M2")
        
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n🎉 Training completed in {training_time/60:.1f} minutes!")
        
        # Save model
        print("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(config["output_dir"])
        
        # Save training info
        training_info = {
            "model_name": config["model_name"],
            "training_examples": len(texts),
            "epochs": config["num_epochs"],
            "training_time_minutes": training_time / 60,
            "device": device,
            "status": "completed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(config["output_dir"], "training_info.json"), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"✅ Model saved to: {config['output_dir']}")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_phi2_training()
    if success:
        print("\n🎉 Phi-2 training completed successfully!")
    else:
        print("\n❌ Phi-2 training failed")