"""Inference engine for domain name generation"""
import torch
import re
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

from ..utils.config import Config

class DomainGenerator:
    """Generate domain name suggestions using fine-tuned models"""
    
    def __init__(self, model_path: str, base_model_name: str, config: Config) -> None:
        self.config = config
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = config.device
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the fine-tuned model and tokenizer"""
        print(f"Loading model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device != "mps" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        if self.device == "mps":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Model loaded successfully")
    
    def create_prompt(self, business_description: str, target_audience: Optional[str] = None) -> str:
        """Create a formatted prompt for domain generation"""
        
        prompt = f"""Generate 5 professional domain name suggestions for the following business:

Business Description: {business_description}"""
        
        if target_audience:
            prompt += f"\nTarget Audience: {target_audience}"
        
        prompt += """

Requirements:
- Domain names should be memorable and relevant
- Keep domains between 6-15 characters when possible  
- Avoid generic terms like "business" or "company"
- Prefer .com, .co, .io, or industry-relevant extensions
- Ensure names are professional and brandable

Domain suggestions:"""
        
        return prompt
    
    def generate_domains(
        self, 
        business_description: str, 
        target_audience: Optional[str] = None,
        num_suggestions: int = 5,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_length: Optional[int] = None
    ) -> List[str]:
        """Generate domain name suggestions using the fine-tuned model.
        
        Args:
            business_description: Description of the business needing domain names
            target_audience: Optional target audience description
            num_suggestions: Number of domain suggestions to return
            temperature: Sampling temperature for generation
            top_p: Top-p sampling parameter
            max_length: Maximum sequence length
            
        Returns:
            List of clean, validated domain name suggestions
        """
        
        # Use config defaults if not specified
        temperature = temperature or self.config.model.temperature
        top_p = top_p or self.config.model.top_p
        max_length = max_length or self.config.model.max_length
        
        # Create prompt
        prompt = self.create_prompt(business_description, target_audience)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length - 200  # Leave room for generation
        )
        
        if self.device == "mps":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=temperature,
                top_p=top_p,
                top_k=self.config.model.top_k,
                do_sample=self.config.model.do_sample,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract domain suggestions from response
        domains = self._extract_domains(full_response, prompt)
        
        # Clean and validate domains
        clean_domains = self._clean_and_validate_domains(domains)
        
        return clean_domains[:num_suggestions]
    
    def _extract_domains(self, response: str, prompt: str) -> List[str]:
        """Extract domain names from model response"""
        
        # Remove the prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        domains = []
        
        # Pattern to match numbered list items
        list_pattern = r'^\d+\.\s*(.+)$'
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for numbered list format
            match = re.match(list_pattern, line)
            if match:
                domain = match.group(1).strip()
                domains.append(domain)
            else:
                # Check if line looks like a domain
                if self._looks_like_domain(line):
                    domains.append(line)
        
        return domains
    
    def _looks_like_domain(self, text: str) -> bool:
        """Check if text looks like a domain name"""
        # Basic domain pattern
        domain_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$'
        
        # Remove common prefixes/suffixes
        text = text.strip().lower()
        text = re.sub(r'^[-•*]\s*', '', text)  # Remove bullet points
        
        return bool(re.match(domain_pattern, text))
    
    def _clean_and_validate_domains(self, domains: List[str]) -> List[str]:
        """Clean and validate domain suggestions"""
        clean_domains = []
        
        for domain in domains:
            # Clean the domain
            clean_domain = self._clean_domain(domain)
            
            # Validate domain
            if self._is_valid_domain(clean_domain):
                clean_domains.append(clean_domain)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_domains = []
        for domain in clean_domains:
            if domain.lower() not in seen:
                seen.add(domain.lower())
                unique_domains.append(domain)
        
        return unique_domains
    
    def _clean_domain(self, domain: str) -> str:
        """Clean a domain name string"""
        # Remove common prefixes and formatting
        domain = domain.strip()
        domain = re.sub(r'^[-•*]\s*', '', domain)  # Remove bullet points
        domain = re.sub(r'^\d+\.\s*', '', domain)  # Remove numbering
        domain = domain.replace('"', '').replace("'", "")  # Remove quotes
        
        # Extract just the domain if there's extra text
        words = domain.split()
        for word in words:
            if '.' in word and len(word) > 3:
                return word.lower()
        
        return domain.lower()
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Validate domain name format and quality"""
        if not domain:
            return False
        
        # Basic format validation
        domain_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?\.[a-zA-Z]{2,}$'
        if not re.match(domain_pattern, domain):
            return False
        
        # Length validation
        name_part = domain.split('.')[0]
        if len(name_part) < 2 or len(name_part) > 30:
            return False
        
        # Quality checks
        if self._is_low_quality_domain(domain):
            return False
        
        return True
    
    def _is_low_quality_domain(self, domain: str) -> bool:
        """Check if domain is low quality"""
        name_part = domain.split('.')[0]
        
        # Check for generic terms
        generic_terms = ['business', 'company', 'corp', 'inc', 'llc', 'ltd']
        if any(term in name_part for term in generic_terms):
            return True
        
        # Check for excessive numbers
        if sum(c.isdigit() for c in name_part) > len(name_part) // 2:
            return True
        
        # Check for too many hyphens
        if name_part.count('-') > 2:
            return True
        
        # Check for repetitive patterns
        if len(set(name_part)) < len(name_part) // 3:
            return True
        
        return False
    
    def generate_with_confidence(
        self, 
        business_description: str, 
        target_audience: Optional[str] = None,
        num_suggestions: int = 5
    ) -> List[Dict[str, float]]:
        """Generate domain names with confidence scores.
        
        Args:
            business_description: Description of the business
            target_audience: Optional target audience
            num_suggestions: Number of suggestions to return
            
        Returns:
            List of dictionaries with 'domain' and 'confidence' keys
        """
        
        domains = self.generate_domains(
            business_description, 
            target_audience, 
            num_suggestions
        )
        
        # Calculate basic confidence scores
        results = []
        for i, domain in enumerate(domains):
            confidence = self._calculate_confidence(domain, business_description)
            results.append({
                "domain": domain,
                "confidence": confidence
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return results
    
    def _calculate_confidence(self, domain: str, business_description: str) -> float:
        """Calculate confidence score for a domain suggestion"""
        score = 0.0
        
        # Length score (prefer 6-12 characters)
        name_part = domain.split('.')[0]
        length = len(name_part)
        if 6 <= length <= 12:
            score += 0.3
        elif 4 <= length <= 15:
            score += 0.2
        else:
            score += 0.1
        
        # TLD score (prefer common TLDs)
        tld = domain.split('.')[-1]
        if tld in ['com', 'co', 'io']:
            score += 0.2
        elif tld in ['net', 'org', 'app']:
            score += 0.15
        else:
            score += 0.1
        
        # Keyword relevance (basic check)
        business_words = re.findall(r'\b\w+\b', business_description.lower())
        domain_words = re.findall(r'\b\w+\b', name_part.lower())
        
        common_words = set(business_words) & set(domain_words)
        if common_words:
            score += 0.3
        
        # Memorability (no complex patterns)
        if not re.search(r'\d{3,}', name_part) and name_part.count('-') <= 1:
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0

class ModelComparator:
    """Compare multiple domain generation models"""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.models = {}
    
    def add_model(self, name: str, model_path: str, base_model_name: str) -> None:
        """Add a model for comparison"""
        self.models[name] = DomainGenerator(model_path, base_model_name, self.config)
    
    def compare_models(
        self, 
        business_descriptions: List[str], 
        num_suggestions: int = 5
    ) -> Dict[str, List[Dict]]:
        """Compare all models on a set of business descriptions"""
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Generating with {model_name}...")
            model_results = []
            
            for description in business_descriptions:
                suggestions = model.generate_with_confidence(
                    description, 
                    num_suggestions=num_suggestions
                )
                
                model_results.append({
                    "business_description": description,
                    "suggestions": suggestions
                })
            
            results[model_name] = model_results
        
        return results
    
    def save_comparison_results(self, results: Dict, output_path: str) -> None:
        """Save comparison results to file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Comparison results saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    config = Config()
    
    # Create domain generator with M1-optimized model
    generator = DomainGenerator(
        model_path="models/dialogpt-medium-domain-generator/final",
        base_model_name="microsoft/DialoGPT-medium",
        config=config
    )
    
    # Test generation
    business_desc = "innovative AI-powered restaurant management platform for small businesses"
    suggestions = generator.generate_with_confidence(business_desc)
    
    print(f"Business: {business_desc}")
    print("Domain suggestions:")
    for suggestion in suggestions:
        print(f"  {suggestion['domain']} (confidence: {suggestion['confidence']:.2f})")