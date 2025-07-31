#!/usr/bin/env python3
"""Simple API test for domain name generation with JSON input/output"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from domain_generator.models.jupyter_compatible import create_generator
from domain_generator.safety.content_filter import ComprehensiveSafetyFilter

class DomainGeneratorAPI:
    """Simple API interface for domain generation testing"""
    
    def __init__(self, model_name: str = "llama-3.2-1b"):
        """Initialize API with specified model"""
        self.model_name = model_name
        self.generator = create_generator(model_name)
        self.safety_filter = ComprehensiveSafetyFilter()
        print(f"✅ API initialized with {model_name}")
    
    def generate_domains(self, request: dict) -> dict:
        """
        Generate domain suggestions from JSON request
        
        Args:
            request: {"business_description": "description here"}
            
        Returns:
            {"suggestions": [{"domain": "...", "confidence": 0.xx}], "status": "..."}
        """
        try:
            # Validate input
            if not isinstance(request, dict) or "business_description" not in request:
                return {
                    "suggestions": [],
                    "status": "error",
                    "message": "Invalid request format. Expected: {'business_description': 'text'}"
                }
            
            business_description = request["business_description"]
            
            # Safety check
            safety_result = self.safety_filter.filter_content(business_description)
            if not safety_result.is_safe:
                return {
                    "suggestions": [],
                    "status": "blocked",
                    "message": f"Request contains inappropriate content: {safety_result.blocked_reason}"
                }
            
            # Generate domains (mock for demo since models aren't trained)
            domains = self._generate_mock_domains(business_description)
            
            # Format response
            suggestions = []
            for i, domain in enumerate(domains):
                # Mock confidence scores (decreasing)
                confidence = round(0.95 - (i * 0.05), 2)
                suggestions.append({
                    "domain": domain,
                    "confidence": confidence
                })
            
            return {
                "suggestions": suggestions,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "suggestions": [],
                "status": "error",
                "message": f"Generation failed: {str(e)}"
            }
    
    def _generate_mock_domains(self, business_description: str) -> list:
        """Generate mock domains for testing (simulates trained model output)"""
        import re
        import random
        
        # Extract keywords
        words = re.findall(r'\b\w+\b', business_description.lower())
        keywords = [w for w in words if len(w) > 3 and w not in ['the', 'and', 'for', 'with', 'that', 'this']]
        
        domains = []
        tlds = ['.com', '.co', '.io', '.org', '.net']
        
        if not keywords:
            return ["domain1.com", "domain2.co", "domain3.io"]
        
        # Generate realistic domain suggestions
        primary_keyword = keywords[0] if keywords else "business"
        
        strategies = [
            lambda k: f"{k}hub.com",
            lambda k: f"my{k}.co", 
            lambda k: f"{k}pro.io",
            lambda k: f"the{k}.com",
            lambda k: f"{k}zone.net"
        ]
        
        for i, strategy in enumerate(strategies):
            if i < len(keywords):
                domain = strategy(keywords[i])
            else:
                domain = strategy(primary_keyword)
            domains.append(domain)
        
        return domains[:5]

def run_api_tests():
    """Run comprehensive API tests"""
    print("🧪 Domain Generator API Tests")
    print("=" * 40)
    
    # Test both models
    models_to_test = ["llama-3.2-1b", "phi-3-mini"]
    
    test_cases = [
        {
            "name": "Valid Business Request",
            "request": {"business_description": "organic coffee shop in downtown area"},
            "expected_status": "success"
        },
        {
            "name": "Tech Startup Request", 
            "request": {"business_description": "AI-powered fitness tracking mobile app"},
            "expected_status": "success"
        },
        {
            "name": "Safety Block Test",
            "request": {"business_description": "adult content website with explicit nude content"},
            "expected_status": "blocked"
        },
        {
            "name": "Invalid Request Format",
            "request": {"invalid_field": "test"},
            "expected_status": "error"
        },
        {
            "name": "Empty Description",
            "request": {"business_description": ""},
            "expected_status": "success"  # Should still work but with generic domains
        }
    ]
    
    for model_name in models_to_test:
        print(f"\n🤖 Testing {model_name}")
        print("-" * 30)
        
        try:
            api = DomainGeneratorAPI(model_name)
            
            for test_case in test_cases:
                print(f"\n📝 Test: {test_case['name']}")
                print(f"Request: {json.dumps(test_case['request'])}")
                
                start_time = time.time()
                response = api.generate_domains(test_case['request'])
                end_time = time.time()
                
                print(f"Response: {json.dumps(response, indent=2)}")
                print(f"Status: {'✅' if response['status'] == test_case['expected_status'] else '❌'} Expected {test_case['expected_status']}, got {response['status']}")
                print(f"Time: {(end_time - start_time)*1000:.0f}ms")
                
        except Exception as e:
            print(f"❌ Model {model_name} failed: {e}")
    
    print(f"\n✅ API Tests Complete!")

def interactive_test():
    """Interactive testing mode"""
    print("🎯 Interactive Domain Generator Test")
    print("=" * 35)
    print("Enter business descriptions to test the API")
    print("Type 'quit' to exit")
    
    api = DomainGeneratorAPI("llama-3.2-1b")
    
    while True:
        try:
            business_desc = input("\n💼 Business description: ").strip()
            
            if business_desc.lower() in ['quit', 'exit', 'q']:
                break
                
            if not business_desc:
                print("Please enter a business description")
                continue
            
            request = {"business_description": business_desc}
            
            start_time = time.time()
            response = api.generate_domains(request)
            end_time = time.time()
            
            print(f"\n📤 Request:")
            print(json.dumps(request, indent=2))
            
            print(f"\n📥 Response:")
            print(json.dumps(response, indent=2))
            
            print(f"\n⏱️  Response time: {(end_time - start_time)*1000:.0f}ms")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Domain Generator API")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--model", "-m", default="llama-3.2-1b",
                       choices=["llama-3.2-1b", "phi-3-mini"],
                       help="Model to test")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_test()
    else:
        run_api_tests()