# Domain Name Generator - Setup Instructions

**NEW**: Now features Phi-2 and Mistral 7B models with 1-epoch training, real-time tqdm progress bars, and Jupyter notebook support.

## 🚀 Ultra-Quick Start (1 Epoch Training)

```bash
# Clone repository
git clone https://github.com/Paulino-Cristovao/domain-name-generator.git
cd domain-name-generator

# Install core dependencies (no API keys needed)
pip install torch transformers peft accelerate datasets tqdm

# Option 1: Google Colab (Recommended - Zero setup)
# Open domain_generator_colab_open_access.ipynb in Google Colab

# Option 2: Local training with progress bars
python run_training_demo.py

# Option 3: Jupyter notebook
jupyter notebook domain_generator_local_training.ipynb
```

## 📋 Prerequisites

### System Requirements (Updated)
- **Mac M1/M2** (optimized) or **NVIDIA GPU** with 8GB+ VRAM
- **Python 3.9+**
- **8GB+ RAM** for training
- **5GB+ disk space** for models (reduced with 4-bit quantization)

### Training Times (1 Epoch)
- **Phi-2**: 15-20 minutes
- **Mistral 7B**: 30-45 minutes  
- **Both models**: ~45-65 minutes total

### Optional API Keys
- **OpenAI API Key** (for advanced evaluation only)
- **Weights & Biases API Key** (for experiment tracking)

## 🔧 Installation

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create `.env` file in project root:

```bash
# Weights & Biases Configuration
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=domain-name-generator
WANDB_ENTITY=your-username

# OpenAI Configuration (for evaluation)
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
DEVICE=mps  # Use M1 GPU acceleration

# Safety settings
ENABLE_SAFETY_FILTERS=true
SAFETY_THRESHOLD=0.8
```

### 3. Directory Structure

The installation will create:

```
domain-name-generator/
├── data/
│   ├── processed/          # Generated datasets
│   └── results/            # Evaluation results
├── models/
│   ├── tracking/           # Experiment tracking
│   └── [model-variants]/   # Trained models
├── logs/                   # Training logs
├── notebooks/              # Jupyter experiments
├── src/                    # Source code
└── scripts/                # Utility scripts
```

## 🚀 Usage Guide

### Phase 1: Dataset Creation

```bash
# Open Jupyter
jupyter notebook

# Run dataset creation notebook
# notebooks/01_dataset_creation.ipynb
```

This generates:
- `data/processed/training_dataset.json` (1000 training examples)
- `data/processed/edge_cases.json` (200 edge cases)

### Phase 2: Model Training

```bash
# Run training experiments notebook
# notebooks/02_model_training_experiments.ipynb
```

Available models (all <4GB, M1-optimized):
- **Llama-3.2-1B**: 1B params (~3.5GB) - *Default*
- **Phi-3-Mini**: 3.8B params (~3.8GB) 
- **DialoGPT-Medium**: 355M params (~1.4GB)
- **GPT2-Small**: 124M params (~500MB)
- **DistilGPT2**: 82M params (~330MB)

### Phase 3: Model Evaluation

```bash
# Run evaluation framework notebook
# notebooks/03_model_evaluation_framework.ipynb
```

Features:
- GPT-4o LLM-as-a-Judge evaluation
- Edge case discovery
- Performance comparison
- Improvement suggestions

## 🧪 Testing & Validation

### Run Full Test Suite

```bash
python scripts/test_setup.py
```

This validates:
- ✅ Environment setup
- ✅ Package imports
- ✅ Data generation
- ✅ Safety filters
- ✅ Model configurations
- ✅ OpenAI judge connectivity
- ✅ M1 GPU acceleration

### M1 Performance Validation

```bash
python -c \"
import torch
print('MPS Available:', torch.backends.mps.is_available())
device = torch.device('mps')
x = torch.randn(1000, 1000, device=device)
y = torch.mm(x, x.t())
print('M1 GPU Test: Success')
\"
```

## 🎓 Model Training

### Using Jupyter Interface

```python
from src.domain_generator.models.jupyter_compatible import create_generator

# Create generator
generator = create_generator('llama-3.2-1b')  # or 'phi-3-mini'

# Train model (2 epochs, M1 optimized)
model_path = generator.train_model()

# Load trained model
generator.load_model(model_path)

# Generate domains
domains = generator.generate_domains("AI-powered coffee shop")
```

### Using Command Line

```bash
# Reproduce specific experiment
python reproduce_experiment.py llama-3.2-1b

# Custom training
python scripts/train_model.py --model llama-3.2-1b --epochs 2 --batch-size 2
```

## 📊 Evaluation

### Quick Evaluation

```python
from src.domain_generator.models.jupyter_compatible import create_generator

generator = create_generator('llama-3.2-1b')
# Benchmark model performance
metrics = generator.benchmark_model([
    \"AI coffee shop\",
    \"eco fashion brand\", 
    \"VR fitness studio\"
])
```

### Full Evaluation Pipeline

```python
# Load evaluation framework
from src.domain_generator.evaluation.openai_judge import EvaluationFramework
from src.domain_generator.utils.config import Config

config = Config()
framework = EvaluationFramework(config)

# Run comprehensive evaluation
results = await framework.evaluate_model_output(model_name, test_cases)
```

## 🔄 Reproducibility

### Fixed Configuration

All experiments use:
- **Random seeds**: 42 (PyTorch, NumPy, Python)
- **Model versions**: Pinned in requirements.txt
- **Hardware**: M1 MPS acceleration
- **Training epochs**: 2 (all models)

### Reproduction Scripts

```bash
# Reproduce any experiment
python reproduce_experiment.py <model-config-id>

# Example
python reproduce_experiment.py llama-3.2-1b
```

### Version Tracking

- Experiment IDs: `{model}_{timestamp}_{config_hash}`
- Full config logging in `models/tracking/experiments.json`
- Reproducibility info in `models/tracking/reproducibility_info.json`

## 🛠️ Troubleshooting

### Common Issues

**1. MPS Not Available**
```bash
# Check macOS version (requires 12.3+)
sw_vers

# Reinstall PyTorch with MPS support
pip uninstall torch
pip install torch torchvision torchaudio
```

**2. Memory Issues**
```bash
# Use smaller model
generator = create_generator('distilgpt2')  # Only 330MB

# Reduce batch size
# Edit model config or use gradient accumulation
```

**3. OpenAI API Issues**
```bash
# Verify API key
python -c \"import openai; client = openai.OpenAI(); print('API Key valid')\"

# Check rate limits
# Use smaller batch sizes in evaluation
```

**4. Module Import Errors**
```bash
# Add src to Python path
export PYTHONPATH=\"$PYTHONPATH:$(pwd)/src\"

# Or in notebook
import sys
sys.path.append('../src')
```

### Performance Optimization

**M1 Optimization Checklist:**
- ✅ MPS device enabled
- ✅ bfloat16 precision for Llama/Phi
- ✅ Memory-efficient batch sizes
- ✅ Gradient accumulation
- ✅ Model-specific attention modules

**Memory Usage:**
```bash
# Monitor memory during training
python -c \"
import psutil
print(f'Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')
\"
```

## 📚 Advanced Usage

### Custom Model Configuration

```python
from src.domain_generator.models.trainer import create_model_configs
from src.domain_generator.utils.config import LoRAConfig, TrainingConfig

# Create custom config
custom_config = {
    \"model_name\": \"your-model-name\",
    \"lora_config\": LoRAConfig(r=16, lora_alpha=32),
    \"training_config\": TrainingConfig(num_epochs=2, per_device_train_batch_size=2)
}
```

### Batch Processing

```python
# Process multiple business descriptions
business_list = [\"coffee shop\", \"tech startup\", \"fitness center\"]

for business in business_list:
    domains = generator.generate_domains(business)
    print(f\"{business}: {domains[:3]}\")
```

### Integration with External Systems

```python
# Domain availability checking (requires external API)
def check_availability(domain):
    # Implement your domain availability check
    return True

# Filter available domains
available_domains = [d for d in domains if check_availability(d)]
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Run tests: `python scripts/test_setup.py`
4. Submit pull request

## 📄 License

See LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check troubleshooting section above
2. Review test results: `python scripts/test_setup.py`
3. Open GitHub issue with:
   - System info (`sw_vers`, `python --version`)
   - Error messages
   - Reproduction steps

---

**Ready to generate domains! 🚀**