#!/usr/bin/env python3
"""Test script to verify project setup and API connectivity"""
import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        if torch.backends.mps.is_available():
            print("  ✓ MPS (Mac M1 acceleration) available")
        else:
            print("  ⚠️  MPS not available, using CPU")
            
    except ImportError as e:
        print(f"  ❌ PyTorch import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("  ✓ Transformers library")
    except ImportError as e:
        print(f"  ❌ Transformers import failed: {e}")
        return False
    
    try:
        import openai
        print("  ✓ OpenAI library")
    except ImportError as e:
        print(f"  ❌ OpenAI import failed: {e}")
        return False
    
    try:
        from domain_generator.utils.config import Config
        print("  ✓ Domain generator config")
    except ImportError as e:
        print(f"  ❌ Domain generator import failed: {e}")
        return False
    
    return True

def test_data_generation():
    """Test synthetic data generation"""
    print("\n📊 Testing data generation...")
    
    try:
        from domain_generator.data.synthetic_generator import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        
        # Generate a small test dataset
        dataset = generator.generate_training_dataset(num_samples=5)
        print(f"  ✓ Generated {len(dataset)} training examples")
        
        # Test edge cases
        edge_cases = generator.generate_edge_cases(num_cases=10)
        print(f"  ✓ Generated {len(edge_cases)} edge cases")
        
        # Show sample
        sample = dataset[0]
        print(f"  📝 Sample business: {sample['business_description'][:50]}...")
        print(f"  🌐 Sample domains: {sample['good_domains'][:2]}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data generation failed: {e}")
        return False

def test_safety_filters():
    """Test safety filtering system"""
    print("\n🛡️  Testing safety filters...")
    
    try:
        from domain_generator.safety.content_filter import ComprehensiveSafetyFilter
        
        safety_filter = ComprehensiveSafetyFilter()
        
        # Test safe content
        safe_result = safety_filter.filter_content("innovative coffee shop in downtown")
        if safe_result.is_safe:
            print("  ✓ Safe content correctly allowed")
        else:
            print(f"  ⚠️  Safe content blocked: {safe_result.blocked_reason}")
        
        # Test unsafe content
        unsafe_result = safety_filter.filter_content("adult entertainment website with explicit content")
        if not unsafe_result.is_safe:
            print("  ✓ Unsafe content correctly blocked")
        else:
            print("  ⚠️  Unsafe content not blocked")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Safety filter test failed: {e}")
        return False

async def test_openai_judge():
    """Test OpenAI judge connectivity"""
    print("\n⚖️  Testing OpenAI judge...")
    
    try:
        from domain_generator.evaluation.openai_judge import EvaluationFramework
        from domain_generator.utils.config import Config
        
        config = Config()
        framework = EvaluationFramework(config)
        
        # Test with a simple evaluation
        test_case = {
            "business_description": "organic coffee shop",
            "suggestions": ["organicbean.com", "coffeeco123.com"]
        }
        
        print("  🔄 Running test evaluation (this may take a moment)...")
        results = await framework.evaluate_model_output("test_model", [test_case], batch_size=1)
        
        if "error" not in results:
            print(f"  ✅ OpenAI judge working! Test score: {results['overall_score']['mean']:.2f}")
            return True
        else:
            print(f"  ❌ OpenAI judge failed: {results['error']}")
            return False
            
    except Exception as e:
        print(f"  ❌ OpenAI judge test failed: {e}")
        print("  💡 Make sure your OPENAI_API_KEY is set in .env file")
        return False

def test_model_config():
    """Test model configurations"""
    print("\n🤖 Testing model configurations...")
    
    try:
        from domain_generator.models.trainer import create_model_configs
        
        configs = create_model_configs()
        
        for model_name, config in configs.items():
            print(f"  ✓ {model_name}: {config['model_name']}")
            print(f"    LoRA rank: {config['lora_config'].r}")
            print(f"    Batch size: {config['training_config'].per_device_train_batch_size}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model config test failed: {e}")
        return False

def check_environment():
    """Check environment setup"""
    print("\n🌍 Checking environment...")
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("  ✓ .env file exists")
        
        # Check if API key is set
        with open(env_file) as f:
            content = f.read()
            if "OPENAI_API_KEY=" in content and "sk-" in content:
                print("  ✓ OpenAI API key configured")
            else:
                print("  ⚠️  OpenAI API key not found in .env")
                return False
    else:
        print("  ❌ .env file not found")
        return False
    
    # Check required directories
    required_dirs = ["data/processed", "models", "logs"]
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path} directory exists")
        else:
            print(f"  ⚠️  {dir_path} directory missing")
    
    return True

async def main():
    """Run all tests"""
    print("🧪 Domain Name Generator - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Environment", check_environment),
        ("Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Safety Filters", test_safety_filters),
        ("Model Config", test_model_config),
    ]
    
    # Run synchronous tests
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Run async test
    try:
        openai_result = await test_openai_judge()
        results.append(("OpenAI Judge", openai_result))
    except Exception as e:
        print(f"  ❌ OpenAI Judge test crashed: {e}")
        results.append(("OpenAI Judge", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Generate training data: jupyter notebook notebooks/01_dataset_creation.ipynb")
        print("2. Train a model: python scripts/train_model.py --model mistral-7b --dataset data/processed/training_dataset.json")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        
        if not any(name == "OpenAI Judge" and result for name, result in results):
            print("\n💡 Common issues:")
            print("- Make sure OPENAI_API_KEY is set in .env file")
            print("- Check internet connection")
            print("- Verify API key is valid")

if __name__ == "__main__":
    asyncio.run(main())