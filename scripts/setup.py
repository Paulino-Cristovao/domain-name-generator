#!/usr/bin/env python3
"""Setup script for domain generator project"""
import os
import sys
import subprocess
from pathlib import Path
import json

def create_directories():
    """Create necessary project directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/results",
        "models/baseline",
        "models/improved",
        "models/final",
        "logs",
        "outputs"
    ]
    
    print("Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")

def create_env_file():
    """Create .env file template"""
    env_content = """# API Keys for LLM-as-a-Judge evaluation
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Weights & Biases for experiment tracking
WANDB_API_KEY=your_wandb_api_key_here

# Model configurations
DEVICE=mps  # Use 'cuda' for NVIDIA GPUs, 'cpu' for CPU only

# Safety settings
ENABLE_SAFETY_FILTERS=true
SAFETY_THRESHOLD=0.8
"""
    
    env_path = Path(".env.example")
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✓ Created .env.example file")
        print("  Please copy to .env and add your API keys")
    else:
        print("✓ .env.example already exists")

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)
    else:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")

def install_dependencies():
    """Install Python dependencies"""
    print("Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Try installing manually with: pip install -r requirements.txt")
        return False
    
    return True

def test_pytorch_mps():
    """Test if PyTorch MPS is available on Mac M1"""
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✓ PyTorch MPS (Mac M1 acceleration) available")
            return True
        else:
            print("⚠️  PyTorch MPS not available, will use CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def generate_sample_data():
    """Generate sample training data"""
    print("Generating sample training data...")
    
    try:
        sys.path.append('src')
        from domain_generator.data.synthetic_generator import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        
        # Generate small sample dataset for testing
        dataset = generator.generate_training_dataset(num_samples=100)
        
        # Save to processed data directory
        output_path = Path("data/processed/sample_dataset.json")
        generator.save_dataset(dataset, str(output_path))
        
        print(f"✓ Sample dataset generated: {output_path}")
        print(f"  Contains {len(dataset)} training examples")
        
    except Exception as e:
        print(f"⚠️  Failed to generate sample data: {e}")
        print("You can generate data later using the notebooks")

def test_safety_filters():
    """Test safety filtering components"""
    print("Testing safety filters...")
    
    try:
        sys.path.append('src')
        from domain_generator.safety.content_filter import ComprehensiveSafetyFilter
        
        safety_filter = ComprehensiveSafetyFilter()
        
        # Test with safe content
        safe_result = safety_filter.filter_content("innovative coffee shop in downtown")
        
        # Test with unsafe content  
        unsafe_result = safety_filter.filter_content("adult entertainment website")
        
        if safe_result.is_safe and not unsafe_result.is_safe:
            print("✓ Safety filters working correctly")
        else:
            print("⚠️  Safety filters may need attention")
            
    except Exception as e:
        print(f"⚠️  Safety filter test failed: {e}")

def create_sample_config():
    """Create sample configuration files"""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # Training config
    training_config = {
        "model": {
            "mistral-7b": {
                "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
                "max_length": 512,
                "temperature": 0.7
            },
            "phi-3.5-mini": {
                "model_name": "microsoft/Phi-3.5-mini-instruct", 
                "max_length": 512,
                "temperature": 0.7
            }
        },
        "training": {
            "num_epochs": 3,
            "batch_size": 1,
            "learning_rate": 2e-4,
            "save_steps": 500,
            "eval_steps": 500
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1
        }
    }
    
    config_path = config_dir / "training_config.yaml"
    if not config_path.exists():
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(training_config, f, default_flow_style=False)
        print(f"✓ Created {config_path}")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Copy .env.example to .env and add your API keys:")
    print("   cp .env.example .env")
    print("   # Edit .env with your OpenAI API key")
    print()
    print("2. Generate full training dataset:")
    print("   jupyter notebook notebooks/01_dataset_creation.ipynb")
    print()
    print("3. Train your first model:")
    print("   python scripts/train_model.py --model phi-3.5-mini --dataset data/processed/training_dataset.json")
    print()
    print("4. Evaluate the model:")
    print("   python scripts/evaluate_model.py --model models/phi-3.5-mini-domain-generator/final")
    print()
    print("5. Run complete experiments:")
    print("   jupyter notebook notebooks/")
    print()
    print("📚 Documentation: README.md")
    print("🐛 Issues: Create GitHub issues for problems")

def main():
    """Main setup function"""
    print("🚀 Setting up Domain Name Generator project...")
    print()
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Create environment file
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Continuing setup despite dependency installation issues...")
    
    # Test PyTorch MPS
    test_pytorch_mps()
    
    # Generate sample data
    generate_sample_data()
    
    # Test safety filters
    test_safety_filters()
    
    # Create sample config
    create_sample_config()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()