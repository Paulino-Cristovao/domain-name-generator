# 🚀 Training Guide: Phi-2 & Mistral 7B Models (1 Epoch)

## 📋 Training Configurations Created

### 🧠 Phi-2 Model
- **Model**: `microsoft/phi-2` (2.7B parameters)
- **Size**: ~3.2GB (4-bit quantization)
- **Dataset**: 40 training examples
- **Epochs**: 1 (for quick testing)
- **Batch Size**: 1
- **Learning Rate**: 2e-4
- **Config**: `models/phi-2-domain-generator/training_config.json`

### 🌟 Mistral 7B Model
- **Model**: `mistralai/Mistral-7B-Instruct-v0.1` (7B parameters)
- **Size**: ~3.8GB (GPTQ/4-bit quantization)
- **Dataset**: 40 training examples
- **Epochs**: 1 (for quick testing)
- **Batch Size**: 1
- **Learning Rate**: 1e-4 (lower for larger model)
- **Gradient Accumulation**: 16 steps
- **Config**: `models/mistral-7b-domain-generator/training_config.json`

## ⚠️ Environment Issue

The current local environment has numpy compatibility issues that prevent training:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

## 💡 Recommended Training Approaches

### Option 1: Google Colab (Recommended)
1. Open `domain_generator_colab_open_access.ipynb` in Google Colab
2. Run all cells to train both models
3. Both models are configured for 1 epoch training
4. Training will work immediately with no setup required

### Option 2: Fresh Environment
```bash
# Create new conda environment
conda create -n domain-gen python=3.9
conda activate domain-gen

# Install dependencies
pip install torch transformers peft accelerate datasets
pip install scikit-learn pandas numpy matplotlib seaborn
pip install python-dotenv pyyaml tqdm

# Run training
python src/domain_generator/models/trainer.py
```

### Option 3: Docker Environment
```bash
# Use a clean PyTorch container
docker run -it --gpus all pytorch/pytorch:latest bash
```

## 🎯 Training Dataset

**File**: `data/processed/phi2_mistral_training_dataset.json`
**Examples**: 40 (8 unique × 5 duplicates for better training)

**Categories Covered**:
- 🤖 Tech/AI (restaurant management, ML consulting)
- 🛒 E-commerce (eco-friendly clothing, coffee subscription)
- 💪 Health/Fitness (VR fitness, meditation apps)
- 🎓 Education (coding bootcamp, language learning)
- 💰 Finance (crypto trading, accounting software)

## ⏱️ Expected Training Times

### With GPU:
- **Phi-2**: ~15-20 minutes (1 epoch)
- **Mistral 7B**: ~30-45 minutes (1 epoch)

### With CPU Only:
- **Phi-2**: ~2-3 hours (1 epoch)
- **Mistral 7B**: ~6-8 hours (1 epoch)

## 🔧 Training Parameters Optimized For:

- **Memory Efficiency**: Batch size 1, gradient accumulation
- **Quick Testing**: 1 epoch only
- **Quality**: LoRA fine-tuning with optimized target modules
- **Stability**: Fixed tokenization, proper padding

## 📊 Model Comparison After Training

Run the comparison script:
```bash
python scripts/compare_baseline_vs_finetuned.py
```

This will:
1. Test baseline (pre-trained) models
2. Test fine-tuned models
3. Generate performance comparisons
4. Create visualizations
5. Save results to `data/results/`

## 🎯 What to Expect

### Baseline Models:
- May generate generic or inconsistent domain suggestions
- Limited understanding of business context
- Variable quality across different business types

### Fine-tuned Models (1 epoch):
- Better domain format consistency
- Improved business context understanding  
- More relevant TLD choices (.com, .io, .co, .app)
- Still room for improvement with more epochs

## 📈 Next Steps After Training

1. **Evaluate Results**: Compare baseline vs fine-tuned performance
2. **Scale Up**: Add more training data and epochs for better quality
3. **Deploy**: Use the best model for domain generation API
4. **Experiment**: Try different prompts and business categories

## 🔍 Monitoring Training

Both models are configured with:
- ✅ Progress bars (tqdm)
- 📊 Loss logging every 10 steps
- 💾 Model saving every 500 steps
- 📋 TensorBoard logging
- 🎯 Early stopping if needed

---

**Ready to train?** 🚀 
Use Google Colab notebook: `domain_generator_colab_open_access.ipynb`