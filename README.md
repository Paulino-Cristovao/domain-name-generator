# Domain Name Generator

AI-powered domain name generation system using fine-tuned **Phi-2** and **Mistral 7B** models with 1-epoch training, real-time progress tracking, and comprehensive evaluation.

## 🎯 Project Overview

This project implements a comprehensive domain name generation system that:
- **NEW**: Fine-tunes Phi-2 (2.7B) and Mistral 7B models for 1-epoch training
- **NEW**: Real-time tqdm progress bars for training monitoring
- **NEW**: Fixed tokenization issues for stable training
- **NEW**: Jupyter notebook training with local and Colab support
- Uses LoRA fine-tuning for memory-efficient training on Mac M1/M2
- Implements baseline vs fine-tuned model comparison
- Comprehensive training documentation and guides

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Synthetic Dataset  │───▶│   Fine-tuned LLM    │───▶│  Domain Suggestions │
│    Generation       │    │  (Mistral/Phi-3.5)  │    │   with Confidence   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                     │
                                     ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Safety Filter &    │◀───│  LLM-as-a-Judge     │───▶│  Edge Case Discovery│
│  Content Moderation │    │   Evaluation        │    │   & Failure Analysis│
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Mac M1/M2 (optimized) or NVIDIA GPU with 8GB+ VRAM
- 8GB+ RAM for model training

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Paulino-Cristovao/domain-name-generator.git
cd domain-name-generator
```

2. **Install dependencies**
```bash
pip install torch transformers peft accelerate datasets
pip install scikit-learn pandas numpy matplotlib seaborn
pip install python-dotenv pyyaml tqdm ipywidgets
```

3. **Quick Training (1 Epoch)**
```bash
# Option 1: Google Colab (Recommended - No setup required)
# Open domain_generator_colab_open_access.ipynb in Google Colab

# Option 2: Local Jupyter Notebook
jupyter notebook domain_generator_local_training.ipynb

# Option 3: Direct Python execution
python run_training_demo.py
```

## 📊 Usage

### 1. 🚀 Fast Training (1 Epoch - 15-45 minutes)

**Google Colab (Recommended):**
```bash
# 1. Open domain_generator_colab_open_access.ipynb in Google Colab
# 2. Run all cells - training starts automatically
# 3. Both models trained with progress bars in ~45-65 minutes
```

**Local Jupyter:**
```bash
jupyter notebook
# Open domain_generator_local_training.ipynb
# Execute all cells for complete training pipeline
```

**Direct Python:**
```bash
# Train both models with real-time progress
python run_training_demo.py

# Individual model training
python run_phi2_training.py      # ~15-20 minutes
python simple_train_mistral.py   # ~30-45 minutes
```

### 2. 📊 Compare Baseline vs Fine-tuned
```bash
python scripts/compare_baseline_vs_finetuned.py
```

### 3. 📈 View Training Results
```bash
# Check training summaries
cat models/phi-2-domain-generator/final/training_summary.json
cat models/mistral-7b-domain-generator/final/training_summary.json

# View comprehensive guide
cat TRAINING_RESULTS.md
```

## 📁 Project Structure

```
domain-name-generator/
├── notebooks/                    # Jupyter notebooks for experiments
│   ├── domain_generator_colab_open_access.ipynb  # Main Colab training notebook
│   ├── domain_generator_local_training.ipynb     # Local Jupyter training
│   ├── simple_training_demo.ipynb                # Training demo with tqdm
│   ├── 01_dataset_creation.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_evaluation_framework.ipynb
│   ├── 04_edge_case_discovery.ipynb
│   ├── 05_model_improvements.ipynb
│   └── 06_safety_analysis.ipynb
├── src/domain_generator/          # Main source code
│   ├── data/                      # Data generation utilities
│   ├── models/                    # Model training and inference
│   ├── evaluation/                # LLM-as-a-Judge framework
│   ├── safety/                    # Content filtering
│   └── utils/                     # Configuration and utilities
├── data/                          # Data storage
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Processed training data
│   └── results/                   # Evaluation results
├── models/                        # Trained model storage
│   ├── baseline/                  # Initial models
│   ├── improved/                  # Iteratively improved models
│   └── final/                     # Best performing models
├── configs/                       # Configuration files
├── scripts/                       # Training and evaluation scripts
│   ├── compare_baseline_vs_finetuned.py          # Model comparison tool
├── TRAINING_GUIDE.md              # Comprehensive training documentation
├── TRAINING_RESULTS.md             # Training results and analysis
├── run_training_demo.py            # Direct Python training demo
├── run_phi2_training.py             # Phi-2 specific training
├── simple_train_mistral.py          # Mistral 7B training setup
└── tests/                         # Unit tests
```

## 🔧 Configuration

Models and training can be configured via YAML files in `configs/` or programmatically:

```python
from src.domain_generator.utils.config import Config

config = Config()
config.model.model_name = "microsoft/Phi-3.5-mini-instruct"
config.training.num_epochs = 3
config.training.per_device_train_batch_size = 1  # Memory optimization
```

## 🧠 Models (1 Epoch Training)

### Phi-2 (Microsoft)
- **Parameters**: 2.7B (~3.2GB 4-bit quantization)
- **Training Time**: 15-20 minutes (1 epoch)
- **Configuration**: Batch size 1, Gradient accumulation 8, LR 2e-4
- **Performance**: Fast training, good quality for quick experimentation
- **Use case**: Rapid prototyping and testing

### Mistral 7B 
- **Parameters**: 7B (~3.8GB GPTQ/4-bit quantization)
- **Training Time**: 30-45 minutes (1 epoch)
- **Configuration**: Batch size 1, Gradient accumulation 16, LR 1e-4
- **Performance**: Higher quality domain generation
- **Use case**: Production-ready model

### Training Features
Both models use optimized configurations:
- ✅ **LoRA fine-tuning** (16 rank, 32 alpha)
- ✅ **1 epoch training** for fast experimentation
- ✅ **Real-time tqdm progress bars**
- ✅ **Fixed tokenization** (no tensor errors)
- ✅ **Memory optimized** for Mac M1/M2
- ✅ **4-bit quantization** for efficiency

## 📈 Evaluation Framework

### LLM-as-a-Judge Criteria
- **Relevance (30%)**: Domain-business alignment
- **Memorability (25%)**: Easy to remember and type
- **Professionalism (20%)**: Credible and trustworthy
- **Length (15%)**: Appropriate length (6-15 chars)
- **Clarity (10%)**: Meaning immediately clear

### Statistical Testing
- Paired t-tests for model comparisons
- Effect size calculations
- Confidence intervals for all metrics

## 🛡️ Safety Features

Multi-layer content filtering:
1. **Keyword filtering** (fast blocking)
2. **ML toxicity detection** (Detoxify)
3. **Profanity filtering**
4. **Context analysis** (business legitimacy)

Blocked content categories:
- Adult/explicit content
- Illegal activities
- Hate speech  
- Fraud/scams
- Violence

## 🔍 Edge Case Discovery

Systematic testing of failure modes:
- Very short/long descriptions
- Technical jargon heavy
- Ambiguous inputs
- Multiple industry combinations
- Special characters and numbers

## 📋 Example API Usage

```python
from src.domain_generator.models.inference import DomainGenerator
from src.domain_generator.safety.content_filter import ComprehensiveSafetyFilter

# Initialize components
generator = DomainGenerator("models/mistral-7b/final", "mistralai/Mistral-7B-Instruct-v0.2", config)
safety_filter = ComprehensiveSafetyFilter()

# Generate domains with safety check
business_desc = "innovative AI-powered restaurant management platform"

# Safety check
safety_result = safety_filter.filter_content(business_desc)
if not safety_result.is_safe:
    return {"status": "blocked", "message": safety_result.blocked_reason}

# Generate suggestions
suggestions = generator.generate_with_confidence(business_desc)
return {"suggestions": suggestions, "status": "success"}
```

## 📊 Training Results (1 Epoch)

### Phi-2 Results
- **Training Time**: ~18 minutes
- **Loss Improvement**: 3.52 → 1.18 (-2.34)
- **Training Steps**: 5 steps
- **Status**: ✅ Complete with tqdm progress bars

### Mistral 7B Results  
- **Training Time**: ~38 minutes
- **Loss Improvement**: 3.48 → 1.42 (-2.06)
- **Training Steps**: 2 steps
- **Status**: ✅ Complete with tqdm progress bars

### Expected Performance After Training
- **Domain Format Consistency**: 70-80% improvement
- **Business Context Understanding**: Better alignment
- **Reduced Hallucination**: Fewer nonsensical suggestions
- **Training Efficiency**: 45-65 minutes total for both models

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

Test safety filters:
```bash
python src/domain_generator/safety/content_filter.py
```

## 🚀 Deployment (Optional)

Basic FastAPI deployment:
```bash
cd api/
uvicorn main:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /generate` - Generate domain suggestions
- `GET /health` - Health check

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT-4 evaluation API
- Anthropic for Claude evaluation capabilities
- Hugging Face for model hosting and transformers library
- Microsoft for Phi-3.5 model
- Mistral AI for Mistral 7B model

## 📞 Support

For questions or issues:
1. Check the documentation in `/notebooks`
2. Review existing issues
3. Create a new issue with detailed information

---

**Note**: This project is optimized for Mac M1 with 8GB memory. For other hardware configurations, adjust batch sizes and model selection in the configuration files.