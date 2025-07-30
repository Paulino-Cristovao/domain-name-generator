"""Configuration management for domain generator"""
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for model training and inference"""
    model_name: str
    model_type: str  # "mistral" or "phi"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_return_sequences: int = 5
    
@dataclass
class LoRAConfig:
    """LoRA configuration for memory-efficient fine-tuning"""
    r: int = 16
    lora_alpha: int = 32
    target_modules: list = None
    lora_dropout: float = 0.1
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

@dataclass 
class TrainingConfig:
    """Training configuration optimized for Mac M1 8GB"""
    output_dir: str = "./models"
    num_epochs: int = 2
    per_device_train_batch_size: int = 1  # Small for memory efficiency
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Compensate for small batch size
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 2  # Save only top 2 checkpoints
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True  # Memory optimization
    dataloader_pin_memory: bool = False  # Disable for Mac M1
    gradient_checkpointing: bool = True  # Memory optimization
    
@dataclass
class EvaluationConfig:
    """Configuration for LLM-as-a-judge evaluation"""
    judge_model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 500
    temperature: float = 0.0  # Deterministic for evaluation
    criteria: Dict[str, float] = None
    
    def __post_init__(self):
        if self.criteria is None:
            self.criteria = {
                "relevance": 0.30,
                "memorability": 0.25, 
                "professionalism": 0.20,
                "length": 0.15,
                "clarity": 0.10
            }

@dataclass
class DatasetConfig:
    """Configuration for synthetic dataset generation"""
    num_samples: int = 1000
    business_types: list = None
    complexity_levels: list = None
    max_domains_per_sample: int = 5
    min_domain_length: int = 4
    max_domain_length: int = 20
    
    def __post_init__(self):
        if self.business_types is None:
            self.business_types = [
                "restaurants", "tech_startups", "creative_agencies",
                "healthcare", "e_commerce", "professional_services",
                "retail", "education", "fitness", "consulting"
            ]
        if self.complexity_levels is None:
            self.complexity_levels = ["simple", "medium", "complex"]

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.model = ModelConfig(
            model_name="meta-llama/Llama-3.2-1B-Instruct",  # 1B params, M1 optimized
            model_type="llama"
        )
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.dataset = DatasetConfig()
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "dataset": self.dataset.__dict__
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @property
    def device(self) -> str:
        """Get appropriate device for Mac M1"""
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

# Global configuration instance
config = Config()