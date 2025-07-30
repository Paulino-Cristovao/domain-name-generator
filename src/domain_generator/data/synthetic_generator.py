"""Synthetic dataset generator for domain name training data"""
import random
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import re

@dataclass
class BusinessDescription:
    """Structure for business description"""
    description: str
    business_type: str
    complexity: str
    keywords: List[str]
    target_audience: str
    location: str = None

class SyntheticDataGenerator:
    """Generate synthetic training data for domain name suggestions"""
    
    def __init__(self):
        self.business_templates = {
            "simple": [
                "{adjective} {business_type} in {location}",
                "{business_type} specializing in {specialty}",
                "Local {business_type} serving {target_audience}",
                "{adjective} {business_type} for {target_audience}"
            ],
            "medium": [
                "{adjective} {business_type} that {service_description} for {target_audience} in {location}",
                "{business_type} combining {concept1} and {concept2} to serve {target_audience}",
                "Innovative {business_type} using {technology} to {service_description}",
                "{adjective} {business_type} focused on {specialty} with {unique_value}"
            ],
            "complex": [
                "{adjective} {business_type} leveraging {technology} to {service_description} for {target_audience} while {additional_benefit}",
                "AI-powered {business_type} that {service_description} using {methodology} to help {target_audience} achieve {goal}",
                "Sustainable {business_type} combining {concept1}, {concept2}, and {concept3} to revolutionize {industry}",
                "{business_type} platform connecting {user_type1} with {user_type2} through {technology} to {service_description}"
            ]
        }
        
        self.vocabulary = {
            "adjectives": [
                "innovative", "modern", "sustainable", "premium", "affordable", "professional",
                "creative", "reliable", "efficient", "cutting-edge", "boutique", "artisan",
                "organic", "eco-friendly", "luxury", "budget-friendly", "expert", "specialized"
            ],
            "business_types": {
                "restaurants": ["restaurant", "cafe", "bistro", "eatery", "diner", "kitchen"],
                "tech_startups": ["startup", "tech company", "software company", "platform", "app", "SaaS"],
                "creative_agencies": ["agency", "studio", "creative firm", "design house", "marketing agency"],
                "healthcare": ["clinic", "practice", "health center", "medical facility", "wellness center"],
                "e_commerce": ["online store", "marketplace", "e-commerce platform", "retail platform"],
                "professional_services": ["consulting firm", "service provider", "advisory", "consultancy"],
                "retail": ["store", "shop", "retailer", "boutique", "outlet"],
                "education": ["school", "academy", "institute", "learning center", "training provider"],
                "fitness": ["gym", "fitness center", "studio", "health club", "training facility"],
                "consulting": ["consultancy", "advisory firm", "consulting group", "professional services"]
            },
            "specialties": [
                "customer experience", "quality products", "personalized service", "innovation",
                "sustainability", "affordability", "expertise", "convenience", "reliability"
            ],
            "technologies": [
                "AI", "machine learning", "blockchain", "IoT", "cloud computing", "data analytics",
                "mobile technology", "automation", "virtual reality", "augmented reality"
            ],
            "target_audiences": [
                "small businesses", "startups", "enterprises", "families", "professionals",
                "students", "seniors", "millennials", "local community", "global clients"
            ],
            "locations": [
                "downtown", "Silicon Valley", "New York", "Los Angeles", "Boston", "Austin",
                "Seattle", "Denver", "Miami", "Chicago", "the Bay Area", "urban areas"
            ],
            "concepts": [
                "technology", "creativity", "sustainability", "innovation", "quality",
                "efficiency", "convenience", "expertise", "community", "wellness"
            ]
        }
        
        self.domain_patterns = {
            "good": [
                "{keyword1}{keyword2}.com",
                "{keyword1}.{tld}",
                "{keyword1}{adjective}.{tld}",
                "{businessname}.{tld}",
                "{keyword1}hub.{tld}",
                "{keyword1}pro.{tld}",
                "my{keyword1}.{tld}",
                "the{keyword1}.{tld}",
                "{keyword1}zone.{tld}",
                "{keyword1}lab.{tld}"
            ],
            "mediocre": [
                "{keyword1}123.{tld}",
                "{keyword1}company.{tld}",
                "{keyword1}business.{tld}",
                "{keyword1}inc.{tld}",
                "{keyword1}corp.{tld}",
                "{location}{keyword1}.{tld}"
            ],
            "bad": [
                "{random_word1}{random_word2}{random_number}.{tld}",
                "business{random_number}.{tld}",
                "company{random_number}.{tld}",
                "{very_long_combination}.{tld}",
                "{typo_keyword}.{tld}",
                "{inappropriate_word}.{tld}"
            ]
        }
        
        self.tlds = [".com", ".net", ".org", ".io", ".co", ".ai", ".app", ".tech"]
        
    def generate_business_description(self, business_type: str, complexity: str) -> BusinessDescription:
        """Generate a single business description"""
        template = random.choice(self.business_templates[complexity])
        
        # Select appropriate business type variations
        business_variations = self.vocabulary["business_types"].get(
            business_type, [business_type]
        )
        
        # Fill template with random vocabulary
        description = template.format(
            adjective=random.choice(self.vocabulary["adjectives"]),
            business_type=random.choice(business_variations),
            specialty=random.choice(self.vocabulary["specialties"]),
            target_audience=random.choice(self.vocabulary["target_audiences"]),
            location=random.choice(self.vocabulary["locations"]),
            service_description=self._generate_service_description(),
            concept1=random.choice(self.vocabulary["concepts"]),
            concept2=random.choice(self.vocabulary["concepts"]),
            concept3=random.choice(self.vocabulary["concepts"]),
            technology=random.choice(self.vocabulary["technologies"]),
            unique_value=random.choice(self.vocabulary["specialties"]),
            additional_benefit=self._generate_additional_benefit(),
            methodology=random.choice(self.vocabulary["technologies"]),
            goal=random.choice(self.vocabulary["specialties"]),
            industry=business_type.replace("_", " "),
            user_type1=random.choice(self.vocabulary["target_audiences"]),
            user_type2=random.choice(self.vocabulary["target_audiences"])
        )
        
        # Extract keywords from description
        keywords = self._extract_keywords(description, business_type)
        
        return BusinessDescription(
            description=description,
            business_type=business_type,
            complexity=complexity,
            keywords=keywords,
            target_audience=random.choice(self.vocabulary["target_audiences"]),
            location=random.choice(self.vocabulary["locations"])
        )
    
    def _generate_service_description(self) -> str:
        """Generate service description phrases"""
        actions = [
            "streamline operations", "enhance productivity", "improve efficiency",
            "deliver quality", "provide solutions", "create value", "drive growth",
            "optimize performance", "enable success", "transform experiences"
        ]
        return random.choice(actions)
    
    def _generate_additional_benefit(self) -> str:
        """Generate additional benefit phrases"""
        benefits = [
            "reducing costs", "saving time", "improving quality", "enhancing security",
            "increasing efficiency", "boosting productivity", "ensuring reliability",
            "maintaining sustainability", "promoting innovation", "building community"
        ]
        return random.choice(benefits)
    
    def _extract_keywords(self, description: str, business_type: str) -> List[str]:
        """Extract relevant keywords from business description"""
        # Remove common words and extract meaningful terms
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "that", "which", "who", "what", "where",
            "when", "why", "how", "is", "are", "was", "were", "be", "being", "been"
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        keywords = [w for w in words if len(w) > 3 and w not in common_words]
        
        # Add business type specific keywords
        business_keywords = {
            "restaurants": ["food", "dining", "eat", "meal", "taste", "chef"],
            "tech_startups": ["tech", "digital", "code", "data", "smart", "app"],
            "healthcare": ["health", "care", "medical", "wellness", "cure", "heal"],
            "e_commerce": ["shop", "buy", "sell", "market", "store", "retail"],
            "fitness": ["fit", "gym", "health", "strong", "active", "train"]
        }
        
        if business_type in business_keywords:
            keywords.extend(random.sample(business_keywords[business_type], 2))
        
        return list(set(keywords[:5]))  # Return unique keywords, limit to 5
    
    def generate_domain_suggestions(self, business: BusinessDescription, quality: str = "good") -> List[str]:
        """Generate domain suggestions based on business description"""
        patterns = self.domain_patterns[quality]
        domains = []
        
        for _ in range(random.randint(3, 5)):
            pattern = random.choice(patterns)
            
            # Generate domain based on pattern
            if quality == "good":
                domain = self._generate_good_domain(pattern, business)
            elif quality == "mediocre":
                domain = self._generate_mediocre_domain(pattern, business)
            else:  # bad
                domain = self._generate_bad_domain(pattern, business)
            
            if domain and len(domain) <= 50:  # Reasonable length limit
                domains.append(domain)
        
        return list(set(domains))  # Remove duplicates
    
    def _generate_good_domain(self, pattern: str, business: BusinessDescription) -> str:
        """Generate high-quality domain names"""
        keywords = business.keywords + [business.business_type.replace("_", "")]
        
        return pattern.format(
            keyword1=random.choice(keywords)[:8],  # Limit length
            keyword2=random.choice(keywords)[:6],
            adjective=random.choice(["pro", "hub", "lab", "zone", "co"])[:4],
            businessname=self._create_business_name(business)[:10],
            tld=random.choice(self.tlds)
        ).lower().replace("_", "").replace(" ", "")
    
    def _generate_mediocre_domain(self, pattern: str, business: BusinessDescription) -> str:
        """Generate mediocre quality domain names"""
        keywords = business.keywords
        
        return pattern.format(
            keyword1=random.choice(keywords)[:8] if keywords else "business",
            location=business.location.replace(" ", "")[:8] if business.location else "local",
            tld=random.choice(self.tlds)
        ).lower().replace("_", "").replace(" ", "")
    
    def _generate_bad_domain(self, pattern: str, business: BusinessDescription) -> str:
        """Generate poor quality domain names"""
        random_words = ["xyz", "abc", "test", "temp", "demo", "sample"]
        random_numbers = [str(random.randint(1, 999)) for _ in range(3)]
        
        # Create intentionally bad domains
        very_long = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=25))
        typo_keyword = business.keywords[0][:-1] + "x" if business.keywords else "busines"
        
        return pattern.format(
            random_word1=random.choice(random_words),
            random_word2=random.choice(random_words),
            random_number=random.choice(random_numbers),
            very_long_combination=very_long,
            typo_keyword=typo_keyword,
            inappropriate_word="badword",  # Placeholder for inappropriate content
            tld=random.choice(self.tlds)
        ).lower()
    
    def _create_business_name(self, business: BusinessDescription) -> str:
        """Create a business name from keywords"""
        if len(business.keywords) >= 2:
            return business.keywords[0][:5] + business.keywords[1][:5]
        elif business.keywords:
            return business.keywords[0][:8] + random.choice(["co", "hub", "pro"])
        else:
            return "mybiz"
    
    def generate_training_dataset(self, num_samples: int = 1000) -> List[Dict]:
        """Generate complete training dataset"""
        dataset = []
        business_types = list(self.vocabulary["business_types"].keys())
        complexities = ["simple", "medium", "complex"]
        
        for i in range(num_samples):
            business_type = random.choice(business_types)
            complexity = random.choice(complexities)
            
            # Generate business description
            business = self.generate_business_description(business_type, complexity)
            
            # Generate positive and negative examples
            good_domains = self.generate_domain_suggestions(business, "good")
            mediocre_domains = self.generate_domain_suggestions(business, "mediocre")
            bad_domains = self.generate_domain_suggestions(business, "bad")
            
            # Create training example
            training_example = {
                "id": i,
                "business_description": business.description,
                "business_type": business.business_type,
                "complexity": business.complexity,
                "keywords": business.keywords,
                "target_audience": business.target_audience,
                "location": business.location,
                "good_domains": good_domains,
                "mediocre_domains": mediocre_domains,
                "bad_domains": bad_domains,
                "prompt": self._create_training_prompt(business),
                "completion": self._create_training_completion(good_domains)
            }
            
            dataset.append(training_example)
        
        return dataset
    
    def _create_training_prompt(self, business: BusinessDescription) -> str:
        """Create training prompt for the model"""
        return f"""Generate 5 professional domain name suggestions for the following business:

Business Description: {business.description}
Target Audience: {business.target_audience}

Requirements:
- Domain names should be memorable and relevant
- Keep domains between 6-15 characters when possible
- Avoid generic terms like "business" or "company"
- Prefer .com, .co, .io, or industry-relevant extensions
- Ensure names are professional and brandable

Domain suggestions:"""
    
    def _create_training_completion(self, good_domains: List[str]) -> str:
        """Create training completion with good domain examples"""
        if not good_domains:
            good_domains = ["example.com"]
        
        completion = "\n"
        for i, domain in enumerate(good_domains[:5], 1):
            completion += f"{i}. {domain}\n"
        
        return completion.strip()
    
    def save_dataset(self, dataset: List[Dict], filepath: str):
        """Save dataset to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
    
    def generate_edge_cases(self, num_cases: int = 100) -> List[Dict]:
        """Generate edge cases for testing model robustness"""
        edge_cases = []
        
        edge_case_types = {
            "very_short": "AI app",
            "very_long": "comprehensive enterprise-level artificial intelligence-powered business process automation and optimization platform for large-scale multinational corporations",
            "technical_jargon": "blockchain-based decentralized autonomous organization utilizing smart contracts for peer-to-peer cryptocurrency transactions",
            "ambiguous": "thing that does stuff for people",
            "multiple_industries": "restaurant and tech startup and consulting firm",
            "special_characters": "café & bistró with 24/7 service",
            "numbers": "24/7 AI-powered automation platform for 365-day operations",
            "acronyms": "B2B SaaS API for CRM and ERP integration"
        }
        
        for case_type, description in edge_case_types.items():
            for i in range(num_cases // len(edge_case_types)):
                business = BusinessDescription(
                    description=description,
                    business_type="edge_case",
                    complexity="complex",
                    keywords=self._extract_keywords(description, "edge_case"),
                    target_audience="various"
                )
                
                edge_case = {
                    "id": f"edge_{case_type}_{i}",
                    "type": case_type,
                    "business_description": description,
                    "prompt": self._create_training_prompt(business),
                    "expected_issues": self._get_expected_issues(case_type)
                }
                
                edge_cases.append(edge_case)
        
        return edge_cases
    
    def _get_expected_issues(self, case_type: str) -> List[str]:
        """Define expected issues for each edge case type"""
        issue_map = {
            "very_short": ["generic_domains", "lack_of_context"],
            "very_long": ["overly_long_domains", "unclear_focus"],
            "technical_jargon": ["incomprehensible_domains", "too_technical"],
            "ambiguous": ["generic_domains", "irrelevant_suggestions"],
            "multiple_industries": ["confused_focus", "generic_terms"],
            "special_characters": ["encoding_issues", "invalid_characters"],
            "numbers": ["number_heavy_domains", "unclear_meaning"],
            "acronyms": ["abbreviation_overuse", "unclear_meaning"]
        }
        
        return issue_map.get(case_type, ["unknown_issues"])

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    
    # Generate training dataset
    print("Generating training dataset...")
    dataset = generator.generate_training_dataset(1000)
    generator.save_dataset(dataset, "data/processed/training_dataset.json")
    
    # Generate edge cases
    print("Generating edge cases...")
    edge_cases = generator.generate_edge_cases(200)
    generator.save_dataset(edge_cases, "data/processed/edge_cases.json")
    
    print(f"Generated {len(dataset)} training examples and {len(edge_cases)} edge cases")