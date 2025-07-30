# Domain Name Generator

AI-powered domain name suggestion system using fine-tuned LLMs with systematic evaluation and improvement cycles.

## 🎯 Project Overview

This project implements a comprehensive domain name generation system that:
- Fine-tunes open-source LLMs (Mistral 7B, Microsoft Phi-3.5 Mini) for domain suggestions
- Uses LLM-as-a-Judge evaluation for systematic quality assessment
- Discovers and addresses edge cases through iterative improvement
- Implements multi-layer safety filtering for content moderation
- Optimized for Mac M1 with 8GB memory using LoRA fine-tuning

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
- Mac M1 with 8GB memory (optimized for)
- OpenAI API key (for LLM-as-a-Judge evaluation)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd domain-name-generator
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
export OPENAI_API_KEY="your-openai-api-key"
```

5. **Create data directories**
```bash
mkdir -p data/{raw,processed,results}
mkdir -p models/{baseline,improved,final}
```

## 📊 Usage

### 1. Generate Training Data
```bash
cd notebooks
jupyter notebook 01_dataset_creation.ipynb
```

### 2. Train Models
```bash
# Train Mistral 7B model
python scripts/train_model.py --model mistral-7b --dataset data/processed/training_dataset.json

# Train Llama 3.1 8B model  
python scripts/train_model.py --model llama-3.1-8b --dataset data/processed/training_dataset.json
```

### 3. Evaluate Models
```bash
python scripts/evaluate_model.py --model-path models/mistral-7b-domain-generator/final --base-model mistralai/Mistral-7B-Instruct-v0.2
```

### 4. Run Complete Experiment Pipeline
```bash
python scripts/run_experiments.py
```

## 📁 Project Structure

```
domain-name-generator/
├── notebooks/                    # Jupyter notebooks for experiments
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

## 🤖 Models

### Mistral 7B
- **Memory efficient**: ~14GB VRAM with LoRA
- **Performance**: Excellent instruction following
- **Use case**: Primary model for production

### Llama 3.1 8B
- **Memory efficient**: ~16GB VRAM with LoRA (fits in Mac M1 with swap)
- **Performance**: State-of-the-art instruction following
- **Use case**: High-quality domain generation

Both models use LoRA fine-tuning with configurations optimized for Mac M1:
- Rank (r): 16-32  
- Alpha: 32-64
- Target modules: Attention layers
- FP16/BF16 mixed precision

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

## 📊 Results

Expected performance targets:
- **Overall Quality Score**: >8.0/10 (vs. baseline ~6.5/10)
- **Edge Case Handling**: <10% failure rate
- **Safety Effectiveness**: >99% inappropriate content blocked
- **Response Time**: <2 seconds for 5 suggestions

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