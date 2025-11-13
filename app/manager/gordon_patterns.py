"""
Gordon Functional Health Patterns Analysis
Detects and analyzes coverage of the 11 Gordon patterns in student conversations.
"""

# Gordon Pattern definitions with keywords
GORDON_PATTERNS = {
    1: {
        "name": "Health Perception / Management",
        "keywords": ["medication", "illness", "doctor", "treatment", "health", "medicine", "sick", "disease", "symptom", "medicijn", "ziekte", "behandeling", "gezondheid", "arts", "dokter"]
    },
    2: {
        "name": "Nutritional–Metabolic",
        "keywords": ["eat", "appetite", "food", "weight", "drink", "meal", "diet", "hungry", "eten", "eetlust", "voedsel", "gewicht", "drinken", "maaltijd", "honger"]
    },
    3: {
        "name": "Elimination",
        "keywords": ["toilet", "urine", "stool", "constipation", "bathroom", "bathroom", "urineren", "ontlasting", "stoelgang", "obstipatie", "wc", "toilet"]
    },
    4: {
        "name": "Activity–Exercise",
        "keywords": ["walk", "move", "exercise", "tired", "active", "activity", "sport", "lopen", "bewegen", "beweging", "oefening", "moe", "actief", "sporten"]
    },
    5: {
        "name": "Sleep–Rest",
        "keywords": ["sleep", "rest", "insomnia", "tired", "sleepy", "bed", "awake", "slaap", "rust", "slapen", "moe", "bed", "wakker", "insomnia"]
    },
    6: {
        "name": "Cognitive–Perceptual",
        "keywords": ["memory", "pain", "vision", "hearing", "think", "understand", "remember", "geheugen", "pijn", "zicht", "horen", "denken", "begrijpen", "herinneren"]
    },
    7: {
        "name": "Self-Perception / Self-Concept",
        "keywords": ["confidence", "sad", "anxious", "self-image", "self-esteem", "feel", "emotion", "vertrouwen", "verdrietig", "angstig", "zelfbeeld", "gevoel", "emotie", "depressief"]
    },
    8: {
        "name": "Role–Relationship",
        "keywords": ["family", "partner", "children", "friends", "relationship", "son", "daughter", "husband", "wife", "familie", "partner", "kinderen", "vrienden", "relatie", "zoon", "dochter"]
    },
    9: {
        "name": "Sexuality–Reproductive",
        "keywords": ["intimacy", "relationship", "partner", "sexual", "romance", "intimiteit", "relatie", "partner", "seksualiteit", "romantisch"]
    },
    10: {
        "name": "Coping–Stress Tolerance",
        "keywords": ["stress", "worry", "anxiety", "coping", "pressure", "overwhelmed", "stress", "zorgen", "angst", "coping", "druk", "overweldigd", "spanning"]
    },
    11: {
        "name": "Values–Belief",
        "keywords": ["religion", "faith", "important", "values", "belief", "meaning", "purpose", "religie", "geloof", "belangrijk", "waarden", "geloof", "betekenis", "doel"]
    }
}


def detect_patterns_in_text(text):
    """
    Detect which Gordon patterns are mentioned in the given text.
    
    Args:
        text: Text to analyze (conversation transcript)
        
    Returns:
        dict: Pattern number -> {
            "name": pattern name,
            "mentioned": bool,
            "keyword_matches": list of matched keywords,
            "mention_count": number of times mentioned
        }
    """
    if not text:
        return {}
    
    text_lower = text.lower()
    pattern_results = {}
    
    for pattern_num, pattern_info in GORDON_PATTERNS.items():
        keyword_matches = []
        mention_count = 0
        
        for keyword in pattern_info["keywords"]:
            if keyword.lower() in text_lower:
                keyword_matches.append(keyword)
                # Count occurrences
                mention_count += text_lower.count(keyword.lower())
        
        pattern_results[pattern_num] = {
            "name": pattern_info["name"],
            "mentioned": len(keyword_matches) > 0,
            "keyword_matches": keyword_matches,
            "mention_count": mention_count
        }
    
    return pattern_results


def analyze_pattern_coverage(conversation_history):
    """
    Analyze Gordon pattern coverage in the conversation.
    
    Args:
        conversation_history: Full conversation text (from read_history)
        
    Returns:
        dict: Analysis results with detected patterns and feedback
    """
    # Extract student messages only (not avatar responses)
    student_messages = []
    for line in conversation_history.split("\n"):
        if line.startswith("Student:"):
            student_text = line.replace("Student:", "").strip()
            if student_text:
                student_messages.append(student_text)
    
    # Combine all student messages
    all_student_text = " ".join(student_messages)
    
    # Detect patterns
    detected_patterns = detect_patterns_in_text(all_student_text)
    
    # Count patterns mentioned
    mentioned_patterns = [num for num, info in detected_patterns.items() if info["mentioned"]]
    total_patterns = len(GORDON_PATTERNS)
    coverage_count = len(mentioned_patterns)
    coverage_percentage = (coverage_count / total_patterns * 100) if total_patterns > 0 else 0
    
    # Generate feedback
    feedback = []
    
    if coverage_count == 0:
        feedback.append("Je hebt geen specifieke gezondheidspatronen besproken. Probeer vragen te stellen over verschillende aspecten van de gezondheid.")
    elif coverage_count <= 3:
        feedback.append(f"Je hebt {coverage_count} gezondheidspatronen besproken. Probeer meer aspecten te verkennen, zoals voeding, slaap, beweging, of ondersteuning.")
    elif coverage_count <= 6:
        feedback.append(f"Goed! Je hebt {coverage_count} gezondheidspatronen besproken. Je toont een breed begrip van de zorgvrager.")
    elif coverage_count <= 9:
        feedback.append(f"Uitstekend! Je hebt {coverage_count} gezondheidspatronen besproken. Je hebt een zeer brede kijk op de situatie van de zorgvrager.")
    else:
        feedback.append(f"Fantastisch! Je hebt {coverage_count} van de 11 gezondheidspatronen besproken. Je hebt een zeer uitgebreide assessment gedaan.")
    
    # Specific pattern feedback
    if mentioned_patterns:
        feedback.append(f"\nBesproken patronen: {', '.join([GORDON_PATTERNS[num]['name'] for num in mentioned_patterns])}")
    
    # Missing important patterns feedback
    important_patterns = [1, 2, 6, 7, 8, 10]  # Health, Nutrition, Cognitive, Self-perception, Relationships, Coping
    missing_important = [num for num in important_patterns if num not in mentioned_patterns]
    
    if missing_important:
        missing_names = [GORDON_PATTERNS[num]['name'] for num in missing_important[:3]]  # Top 3 missing
        feedback.append(f"\nOverweeg ook te vragen naar: {', '.join(missing_names)}")
    
    # Pattern depth feedback (how thoroughly each pattern was discussed)
    thoroughly_discussed = []
    briefly_discussed = []
    
    for pattern_num in mentioned_patterns:
        pattern_info = detected_patterns[pattern_num]
        if pattern_info["mention_count"] >= 3:
            thoroughly_discussed.append(pattern_info["name"])
        elif pattern_info["mention_count"] == 1:
            briefly_discussed.append(pattern_info["name"])
    
    if thoroughly_discussed:
        feedback.append(f"\nJe hebt deze patronen diepgaand besproken: {', '.join(thoroughly_discussed[:3])}")
    
    if briefly_discussed:
        feedback.append(f"\nDeze patronen werden kort aangestipt maar konden meer uitgediept worden: {', '.join(briefly_discussed[:3])}")
    
    summary = " ".join(feedback)
    
    return {
        "total_patterns": total_patterns,
        "covered_patterns": coverage_count,
        "coverage_percentage": round(coverage_percentage, 1),
        "mentioned_patterns": mentioned_patterns,
        "pattern_details": {str(k): v for k, v in detected_patterns.items()},
        "feedback": feedback,
        "summary": summary
    }


def generate_pattern_feedback(conversation_history):
    """
    Main function to generate Gordon pattern analysis feedback.
    
    Args:
        conversation_history: Full conversation text from read_history
        
    Returns:
        dict: Complete pattern analysis with metrics and feedback
    """
    if not conversation_history:
        return {
            "total_patterns": 11,
            "covered_patterns": 0,
            "coverage_percentage": 0,
            "mentioned_patterns": [],
            "pattern_details": {},
            "feedback": ["Geen conversatie om te analyseren."],
            "summary": "Geen conversatie om te analyseren."
        }
    
    return analyze_pattern_coverage(conversation_history)

