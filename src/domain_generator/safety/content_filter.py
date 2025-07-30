"""Safety filtering and content moderation for domain generation"""
import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False
    print("Warning: detoxify not available. Using keyword-based filtering only.")

try:
    from better_profanity import profanity
    PROFANITY_AVAILABLE = True
except ImportError:
    PROFANITY_AVAILABLE = False
    print("Warning: better-profanity not available. Using basic profanity detection.")

@dataclass
class SafetyResult:
    """Result of safety filtering"""
    is_safe: bool
    risk_level: str  # "low", "medium", "high"
    blocked_reason: Optional[str]
    confidence: float
    details: Dict[str, any]

class KeywordFilter:
    """Keyword-based content filtering"""
    
    def __init__(self):
        self.inappropriate_keywords = self._load_inappropriate_keywords()
        self.suspicious_patterns = self._load_suspicious_patterns()
        
    def _load_inappropriate_keywords(self) -> Dict[str, List[str]]:
        """Load categorized inappropriate keywords"""
        return {
            "adult_content": [
                "porn", "sex", "adult", "xxx", "nude", "naked", "escort",
                "massage", "dating", "hookup", "strip", "cam", "webcam"
            ],
            "illegal_activities": [
                "drug", "cocaine", "heroin", "marijuana", "weed", "cannabis",
                "gambling", "casino", "poker", "betting", "lottery",
                "weapon", "gun", "firearm", "explosive", "bomb"
            ],
            "hate_speech": [
                "nazi", "fascist", "terrorist", "supremacist", "racist",
                "bigot", "extremist", "radical", "militant"
            ],
            "fraud_scam": [
                "scam", "fraud", "ponzi", "pyramid", "fake", "counterfeit",
                "phishing", "spam", "mlm", "get rich quick"
            ],
            "violence": [
                "kill", "murder", "assault", "violence", "harm", "hurt",
                "torture", "abuse", "threat", "revenge"
            ]
        }
    
    def _load_suspicious_patterns(self) -> List[str]:
        """Load regex patterns for suspicious content"""
        return [
            r'\b(make|earn)\s+\$\d+\s+(daily|weekly|monthly)\b',  # Get rich quick
            r'\bmoney\s+back\s+guarantee\b',  # Scam indicators
            r'\b(100%|guaranteed)\s+(safe|secure|legal)\b',  # Over-promising
            r'\b(no\s+risk|risk\s+free)\b',  # Unrealistic claims
            r'\b(adult|mature)\s+(content|material|services)\b',  # Adult content
        ]
    
    def check_content(self, text: str) -> SafetyResult:
        """Check text content for inappropriate material"""
        text_lower = text.lower()
        
        # Check for inappropriate keywords
        blocked_categories = []
        for category, keywords in self.inappropriate_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    blocked_categories.append(category)
                    break
        
        # Check for suspicious patterns
        suspicious_patterns_found = []
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text_lower):
                suspicious_patterns_found.append(pattern)
        
        # Determine safety result
        if blocked_categories or suspicious_patterns_found:
            risk_level = "high" if blocked_categories else "medium"
            reason = f"Inappropriate content detected: {', '.join(blocked_categories + suspicious_patterns_found)}"
            
            return SafetyResult(
                is_safe=False,
                risk_level=risk_level,
                blocked_reason=reason,
                confidence=0.9 if blocked_categories else 0.7,
                details={
                    "blocked_categories": blocked_categories,
                    "suspicious_patterns": suspicious_patterns_found
                }
            )
        
        return SafetyResult(
            is_safe=True,
            risk_level="low",
            blocked_reason=None,
            confidence=0.8,
            details={}
        )

class MLContentClassifier:
    """Machine learning-based content classification"""
    
    def __init__(self):
        self.detoxify_model = None
        if DETOXIFY_AVAILABLE:
            try:
                self.detoxify_model = Detoxify('original')
            except Exception as e:
                print(f"Failed to load Detoxify model: {e}")
                self.detoxify_model = None
    
    def check_toxicity(self, text: str) -> SafetyResult:
        """Check text toxicity using ML model"""
        
        if not self.detoxify_model:
            return SafetyResult(
                is_safe=True,
                risk_level="low",
                blocked_reason=None,
                confidence=0.5,
                details={"note": "ML classifier not available"}
            )
        
        try:
            results = self.detoxify_model.predict(text)
            
            # Get highest toxicity score
            max_score = max(results.values())
            max_category = max(results, key=results.get)
            
            # Define thresholds
            high_threshold = 0.8
            medium_threshold = 0.5
            
            if max_score > high_threshold:
                return SafetyResult(
                    is_safe=False,
                    risk_level="high",
                    blocked_reason=f"High toxicity detected: {max_category} ({max_score:.2f})",
                    confidence=max_score,
                    details=results
                )
            elif max_score > medium_threshold:
                return SafetyResult(
                    is_safe=False,
                    risk_level="medium",
                    blocked_reason=f"Moderate toxicity detected: {max_category} ({max_score:.2f})",
                    confidence=max_score,
                    details=results
                )
            else:
                return SafetyResult(
                    is_safe=True,
                    risk_level="low",
                    blocked_reason=None,
                    confidence=1.0 - max_score,
                    details=results
                )
                
        except Exception as e:
            return SafetyResult(
                is_safe=True,
                risk_level="low",
                blocked_reason=None,
                confidence=0.5,
                details={"error": str(e)}
            )

class ProfanityFilter:
    """Profanity detection and filtering"""
    
    def __init__(self):
        self.profanity_available = PROFANITY_AVAILABLE
        if PROFANITY_AVAILABLE:
            profanity.load_censor_words()
    
    def check_profanity(self, text: str) -> SafetyResult:
        """Check for profanity in text"""
        
        if not self.profanity_available:
            # Basic profanity check using simple word list
            basic_profanity = [
                "damn", "hell", "crap", "shit", "fuck", "ass", "bitch",
                "bastard", "bloody", "piss"
            ]
            
            text_lower = text.lower()
            found_profanity = [word for word in basic_profanity if word in text_lower]
            
            if found_profanity:
                return SafetyResult(
                    is_safe=False,
                    risk_level="medium",
                    blocked_reason=f"Profanity detected: {', '.join(found_profanity)}",
                    confidence=0.7,
                    details={"profanity_words": found_profanity}
                )
        else:
            # Use better-profanity library
            if profanity.contains_profanity(text):
                censored = profanity.censor(text)
                profanity_words = [word for word in text.split() if '*' in profanity.censor(word)]
                
                return SafetyResult(
                    is_safe=False,
                    risk_level="medium",
                    blocked_reason=f"Profanity detected in text",
                    confidence=0.8,
                    details={
                        "profanity_words": profanity_words,
                        "censored_text": censored
                    }
                )
        
        return SafetyResult(
            is_safe=True,
            risk_level="low",
            blocked_reason=None,
            confidence=0.8,
            details={}
        )

class ContextAnalyzer:
    """Analyze context and intent of business descriptions"""
    
    def __init__(self):
        self.legitimate_business_indicators = [
            "service", "customer", "quality", "professional", "solution",
            "innovation", "technology", "consulting", "management", "development",
            "software", "platform", "application", "system", "tool",
            "restaurant", "food", "retail", "store", "shop", "market",
            "health", "medical", "fitness", "education", "training"
        ]
        
        self.risk_indicators = [
            "easy money", "no experience required", "work from home",
            "guaranteed income", "instant results", "secret method",
            "limited time", "act now", "urgent", "exclusive offer"
        ]
    
    def analyze_context(self, text: str) -> SafetyResult:
        """Analyze business context for legitimacy"""
        text_lower = text.lower()
        
        # Count legitimate business indicators
        legitimate_count = sum(1 for indicator in self.legitimate_business_indicators 
                             if indicator in text_lower)
        
        # Count risk indicators
        risk_count = sum(1 for indicator in self.risk_indicators 
                        if indicator in text_lower)
        
        # Calculate legitimacy score
        total_words = len(text.split())
        legitimacy_ratio = legitimate_count / max(total_words, 1)
        risk_ratio = risk_count / max(total_words, 1)
        
        # Business description too vague or generic
        if total_words < 5:
            return SafetyResult(
                is_safe=False,
                risk_level="medium",
                blocked_reason="Business description too vague",
                confidence=0.6,
                details={"word_count": total_words}
            )
        
        # High risk indicators
        if risk_ratio > 0.2 or risk_count > 2:
            return SafetyResult(
                is_safe=False,
                risk_level="high",
                blocked_reason="High risk business model detected",
                confidence=0.8,
                details={
                    "risk_indicators": risk_count,
                    "legitimacy_indicators": legitimate_count
                }
            )
        
        # Low legitimacy
        if legitimacy_ratio < 0.1 and legitimate_count == 0:
            return SafetyResult(
                is_safe=False,
                risk_level="medium",
                blocked_reason="Business model unclear or potentially illegitimate",
                confidence=0.7,
                details={
                    "legitimacy_ratio": legitimacy_ratio,
                    "legitimate_indicators": legitimate_count
                }
            )
        
        return SafetyResult(
            is_safe=True,
            risk_level="low",
            blocked_reason=None,
            confidence=0.8,
            details={
                "legitimacy_indicators": legitimate_count,
                "risk_indicators": risk_count
            }
        )

class ComprehensiveSafetyFilter:
    """Multi-layer safety filtering system"""
    
    def __init__(self):
        self.keyword_filter = KeywordFilter()
        self.ml_classifier = MLContentClassifier()
        self.profanity_filter = ProfanityFilter()
        self.context_analyzer = ContextAnalyzer()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def filter_content(self, business_description: str) -> SafetyResult:
        """Comprehensive safety filtering of business description"""
        
        results = []
        
        # Stage 1: Keyword filtering (fast)
        keyword_result = self.keyword_filter.check_content(business_description)
        results.append(("keyword", keyword_result))
        
        # If keyword filter blocks, return immediately for high-risk content
        if not keyword_result.is_safe and keyword_result.risk_level == "high":
            self.logger.warning(f"Content blocked by keyword filter: {keyword_result.blocked_reason}")
            return keyword_result
        
        # Stage 2: Profanity detection
        profanity_result = self.profanity_filter.check_profanity(business_description)
        results.append(("profanity", profanity_result))
        
        # Stage 3: ML toxicity classification
        toxicity_result = self.ml_classifier.check_toxicity(business_description)
        results.append(("toxicity", toxicity_result))
        
        # Stage 4: Context analysis
        context_result = self.context_analyzer.analyze_context(business_description)
        results.append(("context", context_result))
        
        # Combine results
        combined_result = self._combine_safety_results(results)
        
        # Log result
        if not combined_result.is_safe:
            self.logger.warning(f"Content blocked: {combined_result.blocked_reason}")
        else:
            self.logger.info("Content passed safety filters")
        
        return combined_result
    
    def _combine_safety_results(self, results: List[Tuple[str, SafetyResult]]) -> SafetyResult:
        """Combine multiple safety check results"""
        
        # Check if any filter blocked the content
        blocking_results = [(name, result) for name, result in results if not result.is_safe]
        
        if blocking_results:
            # Find highest risk blocking result
            highest_risk = max(blocking_results, key=lambda x: self._risk_level_to_score(x[1].risk_level))
            filter_name, result = highest_risk
            
            return SafetyResult(
                is_safe=False,
                risk_level=result.risk_level,
                blocked_reason=f"Blocked by {filter_name} filter: {result.blocked_reason}",
                confidence=result.confidence,
                details={
                    "blocking_filter": filter_name,
                    "all_results": {name: result.details for name, result in results}
                }
            )
        
        # All filters passed - calculate combined confidence
        confidences = [result.confidence for _, result in results]
        combined_confidence = min(confidences)  # Conservative approach
        
        return SafetyResult(
            is_safe=True,
            risk_level="low",
            blocked_reason=None,
            confidence=combined_confidence,
            details={
                "all_results": {name: result.details for name, result in results}
            }
        )
    
    def _risk_level_to_score(self, risk_level: str) -> int:
        """Convert risk level to numeric score for comparison"""
        scores = {"low": 1, "medium": 2, "high": 3}
        return scores.get(risk_level, 0)
    
    def create_safety_response(self, result: SafetyResult) -> Dict[str, any]:
        """Create API response for safety filtering"""
        
        if result.is_safe:
            return {
                "status": "safe",
                "message": "Business description passed safety checks",
                "risk_level": result.risk_level
            }
        else:
            return {
                "status": "blocked",
                "message": "Request contains inappropriate content",
                "reason": result.blocked_reason,
                "risk_level": result.risk_level,
                "suggestions": []
            }
    
    def test_safety_filters(self) -> Dict[str, bool]:
        """Test all safety filters with known inputs"""
        
        test_cases = {
            "legitimate_business": "innovative AI-powered restaurant management platform",
            "adult_content": "adult entertainment website with explicit content",
            "gambling": "online poker and casino gambling platform",
            "violence": "revenge services for enemies and threats",
            "profanity": "this damn business is fucking awesome",
            "scam": "make $1000 daily with no experience required guaranteed",
            "vague": "thing"
        }
        
        results = {}
        for test_name, test_input in test_cases.items():
            result = self.filter_content(test_input)
            
            # Expected results
            should_be_blocked = test_name != "legitimate_business"
            is_working = (not result.is_safe) == should_be_blocked
            
            results[test_name] = {
                "input": test_input,
                "blocked": not result.is_safe,
                "reason": result.blocked_reason,
                "working_correctly": is_working
            }
        
        return results

# Utility functions
def create_safety_examples() -> List[Dict[str, any]]:
    """Create examples for safety filter testing"""
    
    return [
        {
            "description": "organic coffee shop in downtown area",
            "expected": "safe",
            "category": "legitimate"
        },
        {
            "description": "adult content website with explicit nude content",
            "expected": "blocked",
            "category": "adult_content"
        },
        {
            "description": "online casino and gambling platform",
            "expected": "blocked", 
            "category": "gambling"
        },
        {
            "description": "AI-powered business consulting services",
            "expected": "safe",
            "category": "legitimate"
        },
        {
            "description": "make money fast with no experience required",
            "expected": "blocked",
            "category": "scam"
        }
    ]

if __name__ == "__main__":
    # Test the safety filter
    safety_filter = ComprehensiveSafetyFilter()
    
    print("Testing safety filters...")
    test_results = safety_filter.test_safety_filters()
    
    for test_name, result in test_results.items():
        status = "✓" if result["working_correctly"] else "✗"
        print(f"{status} {test_name}: {result['input']}")
        if not result["working_correctly"]:
            print(f"   Expected blocked, got: {result['blocked']}")
        if result["blocked"]:
            print(f"   Reason: {result['reason']}")
        print()
    
    # Test with examples
    examples = create_safety_examples()
    print("Testing with predefined examples:")
    
    for example in examples:
        result = safety_filter.filter_content(example["description"])
        expected_safe = example["expected"] == "safe"
        actual_safe = result.is_safe
        
        status = "✓" if expected_safe == actual_safe else "✗"
        print(f"{status} {example['category']}: {example['description']}")
        if not actual_safe:
            print(f"   Blocked: {result.blocked_reason}")
        print()