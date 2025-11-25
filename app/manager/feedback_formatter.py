"""
Advanced formatter: produce structured, metric-driven feedback for nursing students.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# Threshold configuration lives here to keep logic easy to tune.
THRESHOLDS = {
    "confidence": {"high": 75, "medium": 45},
    "filler_ratio": {"low": 3, "medium": 7},
    "tempo_variation": {"stable": 10, "high": 20},
    "pause_short_ratio": 55,
    "pause_avg": {"short": 0.5, "long": 2.2},
    "speech_rate": {"slow": 90, "fast": 155},
    "hesitation_markers": {"some": 3, "many": 8},
    "prosody": {"good": 72, "ok": 48},
    "volume_stability": 65,
}

LLM_SECTION_TITLES = [
    "Complimenten",
    "Communicatiegedrag",
    "Gordon-patronen",
    "Klinische redenering",
    "Concrete vervolgstappen",
]

NAME_PATTERN = re.compile(
    r"\b(?:mevrouw|meneer|mw\.|dhr\.|mevr\.|pati[eë]nt)\s+[A-Z][a-z]+(?:\s[A-Z][a-z]+)?",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b")
ADDRESS_PATTERN = re.compile(
    r"\b\d{2,4}\s?(?:[A-Za-z]{2})?\s?(?:straat|laan|weg|dreef|road|rd|st|ave)\b",
    re.IGNORECASE,
)

# Quick reference questions per Gordon-patroon to make follow-ups concrete.
PATTERN_QUESTIONS = {
    # English labels (from GORDON_PATTERNS).
    "Health Perception / Management": "Hoe ervaart u uw gezondheid op dit moment en welke zorg gebruikt u?",
    "Nutritional–Metabolic": "Hoe gaat het met eten en drinken; heeft u genoeg eetlust?",
    "Elimination": "Kunt u vertellen hoe het gaat met plassen en ontlasting?",
    "Activity–Exercise": "Hoe mobiel voelt u zich en wat lukt er in huis qua bewegen?",
    "Sleep–Rest": "Hoe slaapt u de laatste tijd en wordt u uitgerust wakker?",
    "Cognitive–Perceptual": "Merkt u veranderingen in uw geheugen, concentratie of waarneming?",
    "Self-Perception / Self-Concept": "Hoe voelt u zich over uzelf sinds de klachten zijn begonnen?",
    "Role–Relationship": "Wie ondersteunt u thuis en hoe verloopt dat voor u?",
    "Sexuality–Reproductive": "Heeft de situatie invloed op intimiteit of relaties?",
    "Coping–Stress Tolerance": "Wat doet u als het even tegenzit en wat helpt u te ontspannen?",
    "Values–Belief": "Zijn er overtuigingen of waarden die we moeten meenemen in uw zorg?",
    # Dutch labels (defaults in metadata).
    "Gezondheidsbeleving": "Hoe ervaart u uw gezondheid op dit moment en welke zorg gebruikt u?",
    "Voeding": "Hoe gaat het met eten en drinken; heeft u genoeg eetlust?",
    "Uitscheiding": "Kunt u vertellen hoe het gaat met plassen en ontlasting?",
    "Activiteit": "Hoe mobiel voelt u zich en wat lukt er in huis qua bewegen?",
    "Slaap": "Hoe slaapt u de laatste tijd en wordt u uitgerust wakker?",
    "Cognitie": "Merkt u veranderingen in uw geheugen, concentratie of waarneming?",
    "Zelfbeleving": "Hoe voelt u zich over uzelf sinds de klachten zijn begonnen?",
    "Rollen": "Wie ondersteunt u thuis en hoe verloopt dat voor u?",
    "Seksualiteit": "Heeft de situatie invloed op intimiteit of relaties?",
    "Stress": "Wat doet u als het even tegenzit en wat helpt u te ontspannen?",
    "Waarden": "Zijn er overtuigingen of waarden die we moeten meenemen in uw zorg?",
}

# Overall appraisal labels to make tone realistic instead of overly positive.
ASSESSMENT_ICONS = {
    "high": "✅",
    "medium": "⚠️",
    "low": "❌",
}

# Comprehension gap phrases that claim understanding but don't demonstrate it
# Both Dutch and English phrases
# These phrases are INAPPROPRIATE when speaking to someone older - students cannot claim to "understand" them
COMPREHENSION_GAP_PHRASES = [
    # Dutch phrases
    "ik begrijp het",
    "ik snap het",
    "ik begrijp u",
    "ik begrijp je",
    "ik snap u",
    "ik snap je",
    "ja, ik snap het",
    "ja ik snap het",
    "ja, ik begrijp het",
    "ja ik begrijp het",
    "ik weet het",
    "ja, ik weet het",
    "ja ik weet het",
    "oké, duidelijk",
    "oke, duidelijk",
    "duidelijk",
    "oké, ik begrijp het",
    "oke, ik snap het",
    "we gaan het regelen",
    "is goed",
    "ja dat klopt",
    "dat klopt",
    "ja precies",
    "precies",
    "klopt",
    "ja goed",
    "akkoord",
    "ja natuurlijk",
    "natuurlijk",
    "ja inderdaad",
    "inderdaad",
    "ja zo is het",
    "zo is het",
    "ah oké",
    "ah oke",
    "ah duidelijk",
    "oké oké",
    "oke oke",
    # English phrases
    "i understand",
    "i get it",
    "i understand you",
    "yes i understand",
    "yes, i understand",
    "yeah i understand",
    "yeah, i understand",
    "i get it",
    "yes i get it",
    "yes, i get it",
    "yeah i get it",
    "yeah, i get it",
    "got it",
    "yes got it",
    "yes, got it",
    "yeah got it",
    "yeah, got it",
    "i see",
    "yes i see",
    "yes, i see",
    "yeah i see",
    "yeah, i see",
    "yes that's right",
    "that's right",
    "yeah that's right",
    "yes correct",
    "correct",
    "yes exactly",
    "exactly",
    "yeah exactly",
    "yes that's correct",
    "that's correct",
    "yeah that's correct",
    "yes that makes sense",
    "that makes sense",
    "yeah that makes sense",
    "okay i understand",
    "ok i understand",
    "okay got it",
    "ok got it",
    "okay i see",
    "ok i see",
    "sure",
    "yes sure",
    "yes, sure",
    "yeah sure",
    "yeah, sure",
    "alright",
    "yes alright",
    "yes, alright",
    "yeah alright",
    "i know",
    "yes i know",
    "yes, i know",
    "yeah i know",
    "yeah, i know",
    "i know what you mean",
    "i see what you mean",
    "oh i understand",
    "oh i see",
    "ah i see",
    "ah i understand",
    "right i understand",
    "right i get it",
    "makes sense",
    "that makes sense",
]

# Heuristics to spot (perceived) understanding, paraphrasing, and confusion.
UNDERSTANDING_CUES = [
    # Dutch
    "ik begrijp",
    "ik snap",
    "als ik het goed begrijp",
    "als ik u goed begrijp",
    "als ik je goed begrijp",
    "dus u",
    "dus je",
    "dus jij",
    "duidelijk",
    "ok",
    "oke",
    "klinkt duidelijk",
    # English
    "i understand",
    "i get it",
    "i see",
    "if i understand correctly",
    "if i understand",
    "got it",
    "understood",
    "clear",
    "okay",
]
PARAPHRASE_CUES = [
    # Dutch
    "je geeft aan",
    "u geeft aan",
    "wat ik hoor",
    "dus u zegt",
    "dus je zegt",
    "dus u voelt",
    "dus je voelt",
    "dus u bedoelt",
    "met andere woorden",
    "samenvattend",
    "als ik het goed samenvat",
    "kortom",
    "parafrase",
    # English
    "what i hear",
    "so you're saying",
    "so you say",
    "so you feel",
    "in other words",
    "to summarize",
    "if i understand correctly",
    "in summary",
    "what you're saying",
    "you mean",
]
CHECKVRAAG_CUES = [
    # Dutch
    "heb ik dat goed",
    "klopt dat",
    "begrijp ik u goed",
    "heb ik het goed begrepen",
    "is dat correct",
    "is dat juist",
    "vraag",
    "?",
    # English
    "did i understand correctly",
    "is that correct",
    "is that right",
    "have i understood correctly",
    "do i understand correctly",
    "is that accurate",
    "am i correct",
    "did i get that right",
    "did i get it right",
]
CONFUSION_CUES = [
    "ik weet niet",
    "niet zeker",
    "twijfel",
    "lastig",
    "uh",
    "uhm",
    "hmm",
    "even denken",
    "ik ben er niet zeker van",
]

# Conversation analysis rules (strict, scenario-specific)
ANALYSIS_UNDERSTANDING_PHRASES = [
    "ik begrijp het",
    "ik snap het",
    "ik begrijp u",
    "oke duidelijk",
    "oké duidelijk",
    "ja ik snap het",
    "ik weet het",
    "we gaan het regelen",
]
ANALYSIS_FILLER_WORDS = ["eh", "eeh", "ehm", "uh", "uhm", "mmm", "hmm", "emmm"]
ANALYSIS_ABRUPT_CLOSINGS = ["ok dank u", "oke dank u", "oke doei", "ok bye", "we gaan het regelen"]
ANALYSIS_PARAPHRASE_CUES = ["dus u", "als ik het goed begrijp", "bedoelt u dat"]
ANALYSIS_OPEN_QUESTION_PREFIXES = ["wat", "hoe", "waar", "kunt u vertellen"]
ANALYSIS_CLOSED_QUESTION_PREFIXES = ["heb", "is", "kan", "kunt u", "hebt u", "bent u"]
ANALYSIS_SUMMARY_CUES = ["samenvattend", "kortom", "samengevat", "als ik het goed begrijp", "bedoelt u dat", "dus u", "dus je", "dus jij"]
EMPATHY_CUES = ["spijtig", "sorry", "vervelend", "kan me voorstellen", "lijkt lastig", "klinkt moeilijk"]


def scrub_text(text: Optional[str]) -> str:
    """
    Remove potential identifying details from any string content.
    """
    if not text:
        return ""

    cleaned = NAME_PATTERN.sub("[patiënt]", text)
    cleaned = PHONE_PATTERN.sub("[privé-nummer]", cleaned)
    cleaned = ADDRESS_PATTERN.sub("[adres verwijderd]", cleaned)
    cleaned = re.sub(r"\b\d{9}\b", "[id verwijderd]", cleaned)
    return cleaned.strip()


def extract_student_messages(conversation_history: Optional[str]) -> List[str]:
    """
    Collect all student utterances from the conversation log.
    """
    if not conversation_history:
        return []

    messages: List[str] = []
    for line in conversation_history.splitlines():
        if line.strip().startswith("Student:"):
            message = line.split("Student:", 1)[1].strip()
            if message:
                messages.append(message)
    return messages


def split_student_sentences(student_messages: List[str]) -> List[str]:
    """
    Break student messages into sentence-like chunks for token-level scanning.
    """
    sentences: List[str] = []
    for msg in student_messages:
        parts = re.split(r'(?<=[.!?])\s+|\n+', msg)
        for part in parts:
            trimmed = part.strip()
            if trimmed:
                sentences.append(trimmed)
    return sentences


def summarize_filler_tokens(filler_tokens: List[str]) -> str:
    """
    Turn filler tokens into a compact frequency string.
    """
    if not filler_tokens:
        return ""
    counts: Dict[str, int] = {}
    for token in filler_tokens:
        counts[token] = counts.get(token, 0) + 1
    return ", ".join(f"{token} ({counts[token]}x)" for token in sorted(counts.keys()))


def build_conversation_analysis_block(
    conversation_history: Optional[str],
    gordon_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mandatory pre-processing: create a strict transcript-derived analysis block before feedback is generated.
    """
    analysis_block: Dict[str, Any] = {
        "exact_phrases_used": {
            "understanding_phrases": [],
            "filler_words": [],
            "abrupt_closings": [],
            "closed_questions": [],
            "open_questions": [],
            "empathy_attempts": [],
            "paraphrases": [],
        },
        "flags": {
            "comprehension_gap": False,
            "filler_issue": False,
            "low_coverage_issue": False,
            "abrupt_ending_issue": False,
        },
        "extracted_examples": {
            "understanding_examples": [],
            "filler_examples": [],
            "abrubt_ending_examples": [],
        },
        "summary_findings": {
            "pattern_coverage": 0,
            "conversation_length": 0,
            "issue_summary": [],
        },
    }

    pattern_coverage = (gordon_result or {}).get("covered_patterns", 0) or 0
    analysis_block["summary_findings"]["pattern_coverage"] = pattern_coverage
    if pattern_coverage < 3:
        analysis_block["flags"]["low_coverage_issue"] = True

    if not conversation_history:
        if analysis_block["flags"]["low_coverage_issue"]:
            analysis_block["summary_findings"]["issue_summary"].append(
                f"Lage Gordon-dekking: {pattern_coverage}/11 patronen"
            )
        return analysis_block

    student_messages = extract_student_messages(conversation_history)
    if not student_messages:
        if analysis_block["flags"]["low_coverage_issue"]:
            analysis_block["summary_findings"]["issue_summary"].append(
                f"Lage Gordon-dekking: {pattern_coverage}/11 patronen"
            )
        return analysis_block

    sentences = split_student_sentences(student_messages)
    if not sentences:
        if analysis_block["flags"]["low_coverage_issue"]:
            analysis_block["summary_findings"]["issue_summary"].append(
                f"Lage Gordon-dekking: {pattern_coverage}/11 patronen"
            )
        return analysis_block

    analysis_block["summary_findings"]["conversation_length"] = sum(len(s.split()) for s in sentences)
    lower_sentences = [s.lower() for s in sentences]

    paraphrase_indices: List[int] = []
    empathy_attempts: List[str] = []
    summary_indices: set = set()

    # Pass 1: paraphrases, empathy, summaries
    for idx, sentence in enumerate(sentences):
        lower = lower_sentences[idx]
        if any(cue in lower for cue in ANALYSIS_PARAPHRASE_CUES):
            analysis_block["exact_phrases_used"]["paraphrases"].append(sentence)
            paraphrase_indices.append(idx)
        if any(cue in lower for cue in EMPATHY_CUES):
            empathy_attempts.append(sentence)
        if any(cue in lower for cue in ANALYSIS_SUMMARY_CUES):
            summary_indices.add(idx)
    analysis_block["exact_phrases_used"]["empathy_attempts"] = empathy_attempts

    # Pass 2: question types
    for sentence in sentences:
        cleaned = re.sub(r"^[^a-zA-Z0-9]+", "", sentence.strip().lower())
        if not cleaned:
            continue
        if any(cleaned.startswith(prefix + " ") or cleaned == prefix for prefix in ANALYSIS_OPEN_QUESTION_PREFIXES):
            analysis_block["exact_phrases_used"]["open_questions"].append(sentence)
        if any(cleaned.startswith(prefix + " ") or cleaned == prefix for prefix in ANALYSIS_CLOSED_QUESTION_PREFIXES):
            analysis_block["exact_phrases_used"]["closed_questions"].append(sentence)

    # Pass 3: comprehension claims and gaps
    for idx, sentence in enumerate(sentences):
        lower = lower_sentences[idx]
        matched_phrases = [phrase for phrase in ANALYSIS_UNDERSTANDING_PHRASES if phrase in lower]
        for phrase in matched_phrases:
            analysis_block["exact_phrases_used"]["understanding_phrases"].append(phrase)
            window = range(idx, min(len(sentences), idx + 3))
            paraphrase_nearby = any(p_idx in window for p_idx in paraphrase_indices)
            if not paraphrase_nearby:
                analysis_block["flags"]["comprehension_gap"] = True
                if sentence not in analysis_block["extracted_examples"]["understanding_examples"]:
                    analysis_block["extracted_examples"]["understanding_examples"].append(sentence)

    # Pass 4: fillers
    filler_tokens: List[str] = []
    filler_examples: List[str] = []
    for sentence in sentences:
        tokens = re.findall("[\\w']+", sentence.lower())
        sentence_has_filler = False
        for token in tokens:
            if token in ANALYSIS_FILLER_WORDS:
                filler_tokens.append(token)
                sentence_has_filler = True
        if sentence_has_filler:
            filler_examples.append(sentence)

    analysis_block["exact_phrases_used"]["filler_words"] = filler_tokens
    analysis_block["extracted_examples"]["filler_examples"] = filler_examples
    if len(filler_tokens) > 1:
        analysis_block["flags"]["filler_issue"] = True

    # Pass 5: abrupt endings
    abrupt_examples: List[str] = []
    for idx, sentence in enumerate(sentences):
        lower = lower_sentences[idx]
        if any(phrase in lower for phrase in ANALYSIS_ABRUPT_CLOSINGS):
            analysis_block["exact_phrases_used"]["abrupt_closings"].append(sentence)
            nearby_summary = any(
                j in summary_indices or j in paraphrase_indices
                for j in range(max(0, idx - 2), min(len(sentences), idx + 1))
            )
            if not nearby_summary:
                analysis_block["flags"]["abrupt_ending_issue"] = True
                abrupt_examples.append(sentence)
    analysis_block["extracted_examples"]["abrubt_ending_examples"] = abrupt_examples

    # Issue summary
    issues: List[str] = []
    if analysis_block["flags"]["comprehension_gap"]:
        issues.append("Begrip geclaimd zonder parafrase binnen 2 zinnen.")
    if analysis_block["flags"]["filler_issue"]:
        filler_summary = summarize_filler_tokens(filler_tokens)
        issues.append(f"Fillers gedetecteerd: {filler_summary}")
    if analysis_block["flags"]["abrupt_ending_issue"]:
        issues.append("Abrupte afsluiting zonder samenvatting.")
    if analysis_block["flags"]["low_coverage_issue"]:
        issues.append(f"Lage Gordon-dekking: {pattern_coverage}/11 patronen")

    analysis_block["summary_findings"]["issue_summary"] = issues
    return analysis_block


def analyze_understanding_gaps(conversation_history: Optional[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Spot gaps between claimed understanding and demonstrated understanding.
    Detects comprehension gap phrases and checks if they're followed by paraphrases or checkvragen.
    """
    student_messages = extract_student_messages(conversation_history)
    if not student_messages:
        return {
            "gap_detected": False,
            "reasons": [],
            "exact_phrases": [],
            "summary": "Geen studentuitspraken beschikbaar om begrip te toetsen.",
            "student_messages": 0,
            "understanding_statements": 0,
            "paraphrase_attempts": 0,
            "followup_questions": 0,
            "confusion_signals": 0,
        }

    lower_messages = [msg.lower() for msg in student_messages]

    def count_cues(cues: List[str]) -> int:
        return sum(1 for msg in lower_messages for cue in cues if cue in msg)

    def find_exact_phrases(messages: List[str], phrases: List[str]) -> List[Tuple[str, int]]:
        """Find exact phrases used and their message index. Uses flexible matching to catch variations in both Dutch and English."""
        found = []
        found_set = set()  # Track (phrase, idx) to avoid duplicates
        
        for idx, msg in enumerate(messages):
            # Normalize the message: lowercase, remove extra whitespace, normalize punctuation
            msg_lower = msg.lower().strip()
            msg_normalized = re.sub(r'\s+', ' ', msg_lower)  # Multiple spaces to single space
            # Keep punctuation for word boundary matching
            msg_for_matching = msg_normalized
            
            for phrase in phrases:
                # Skip if already found this phrase in this message
                if (phrase, idx) in found_set:
                    continue
                    
                phrase_lower = phrase.lower().strip()
                
                # Strategy 1: Exact match (handles cases where message IS the phrase)
                if phrase_lower == msg_normalized:
                    found.append((phrase, idx))
                    found_set.add((phrase, idx))
                    continue
                
                # Strategy 2: Word boundary matching (most reliable)
                # Create pattern with word boundaries to match whole phrases
                phrase_words = re.split(r'[\s,]+', phrase_lower.strip())
                phrase_pattern = r'\b' + r'\s*[\s,]*'.join([re.escape(word) for word in phrase_words if word]) + r'\b'
                
                if re.search(phrase_pattern, msg_for_matching, re.IGNORECASE):
                    found.append((phrase, idx))
                    found_set.add((phrase, idx))
                    continue
                
                # Strategy 3: Normalized matching (handles punctuation variations)
                # Normalize both: remove punctuation, normalize spaces
                msg_normalized_no_punct = re.sub(r'[^\w\s]', ' ', msg_for_matching)
                msg_normalized_no_punct = re.sub(r'\s+', ' ', msg_normalized_no_punct).strip()
                
                phrase_normalized = re.sub(r'[^\w\s]', ' ', phrase_lower)
                phrase_normalized = re.sub(r'\s+', ' ', phrase_normalized).strip()
                
                # Check if normalized phrase appears as whole words (not substring)
                if phrase_normalized:
                    # Use word boundaries for normalized matching too
                    normalized_pattern = r'\b' + re.escape(phrase_normalized) + r'\b'
                    if re.search(normalized_pattern, msg_normalized_no_punct, re.IGNORECASE):
                        found.append((phrase, idx))
                        found_set.add((phrase, idx))
                        continue
                    
                    # Fallback: check if phrase starts the message
                    if msg_normalized_no_punct.startswith(phrase_normalized + ' ') or msg_normalized_no_punct == phrase_normalized:
                        found.append((phrase, idx))
                        found_set.add((phrase, idx))
        
        return found

    # Find comprehension gap phrases - use improved matching
    gap_phrases_found = find_exact_phrases(student_messages, COMPREHENSION_GAP_PHRASES)
    
    # Debug: log what we found for troubleshooting
    if gap_phrases_found:
        print(f"[DEBUG COMPREHENSION GAP] Found {len(gap_phrases_found)} comprehension gap phrases in messages:")
        for phrase, msg_idx in gap_phrases_found:
            print(f"  - Phrase: '{phrase}' in message {msg_idx}: '{student_messages[msg_idx][:100]}...'")
    
    # Check if each gap phrase is followed by paraphrase or checkvraag
    gap_issues = []
    exact_phrases_quoted = []
    
    for phrase, msg_idx in gap_phrases_found:
        # Check if paraphrase or checkvraag appears in the SAME message or in next 1-2 messages
        followed_by_evidence = False
        
        # First check the same message (after the phrase)
        current_message = student_messages[msg_idx].lower()
        # Try to find phrase position using multiple strategies
        phrase_pos = -1
        phrase_lower = phrase.lower()
        
        # Strategy 1: Direct find
        phrase_pos = current_message.find(phrase_lower)
        
        # Strategy 2: If not found, try normalized matching
        if phrase_pos < 0:
            # Normalize both for matching
            msg_normalized = re.sub(r'[^\w\s]', ' ', current_message)
            msg_normalized = re.sub(r'\s+', ' ', msg_normalized).strip()
            phrase_normalized = re.sub(r'[^\w\s]', ' ', phrase_lower)
            phrase_normalized = re.sub(r'\s+', ' ', phrase_normalized).strip()
            
            if phrase_normalized in msg_normalized:
                # Approximate position - use the normalized phrase start
                phrase_pos = msg_normalized.find(phrase_normalized)
                # Map back to approximate position in original (rough estimate)
                if phrase_pos >= 0:
                    # Count spaces/words before the phrase in normalized version
                    words_before = len(msg_normalized[:phrase_pos].split())
                    # Try to find corresponding position in original
                    words = current_message.split()
                    if words_before < len(words):
                        phrase_pos = len(' '.join(words[:words_before])) if words_before > 0 else 0
        
        if phrase_pos >= 0:
            # Check text after the phrase in the same message
            text_after_phrase = current_message[phrase_pos + len(phrase):]
            has_paraphrase_same = any(cue in text_after_phrase for cue in PARAPHRASE_CUES)
            has_checkvraag_same = any(cue in text_after_phrase for cue in CHECKVRAAG_CUES)
            followed_by_evidence = has_paraphrase_same or has_checkvraag_same
        
        # Also check subsequent messages (within next 2 messages)
        if not followed_by_evidence and msg_idx < len(student_messages) - 1:
            next_messages = student_messages[msg_idx + 1:msg_idx + 3]  # Check next 1-2 messages
            next_lower = " ".join([m.lower() for m in next_messages])
            
            # Check for paraphrase cues
            has_paraphrase = any(cue in next_lower for cue in PARAPHRASE_CUES)
            # Check for checkvraag cues
            has_checkvraag = any(cue in next_lower for cue in CHECKVRAAG_CUES)
            
            followed_by_evidence = has_paraphrase or has_checkvraag
        
        if not followed_by_evidence:
            # This is a comprehension gap - claimed understanding but didn't demonstrate
            exact_phrases_quoted.append(phrase)
            gap_issues.append({
                "phrase": phrase,
                "message_index": msg_idx,
                "exact_message": student_messages[msg_idx]
            })

    understanding_hits = count_cues(UNDERSTANDING_CUES)
    paraphrase_hits = count_cues(PARAPHRASE_CUES)
    checkvraag_hits = count_cues(CHECKVRAAG_CUES)
    confusion_hits = count_cues(CONFUSION_CUES)
    question_count = sum(1 for msg in student_messages if "?" in msg)

    coverage = metadata.get("coverage_percentage", 0)
    missing_patterns = metadata.get("patterns_missing") or []

    gap_reasons: List[str] = []
    
    # Rule 1: Flag comprehension gap phrases - make feedback explicit and directive
    if gap_issues:
        # Group by phrase type
        phrase_counts = {}
        for issue in gap_issues:
            phrase = issue["phrase"]
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Get all unique phrases used
        all_phrases = list(phrase_counts.keys())
        phrases_text = "', '".join(all_phrases[:3])  # Show up to 3 examples
        if len(all_phrases) > 3:
            phrases_text += f" en nog {len(all_phrases) - 3} andere"
        
        total_count = sum(phrase_counts.values())
        
        if total_count == 1:
            gap_reasons.append(f"❌ VERMIJD dit: Je zei '{phrases_text}' tegen iemand die ouder is dan jij. Dit is onbeleefd en onprofessioneel - je kunt niet beweren dat je iemand 'begrijpt' of 'snapt' die ouder is dan jij. Zeg in plaats daarvan NIETS, of gebruik direct een parafrase ('Dus u voelt zich...?') of checkvraag ('Heb ik dat goed begrepen?').")
        else:
            gap_reasons.append(f"❌ VERMIJD dit: Je zei meerdere keren dingen zoals '{phrases_text}' ({total_count} keer) tegen iemand die ouder is dan jij. Dit is onbeleefd en onprofessioneel - je kunt niet beweren dat je iemand 'begrijpt', 'snapt' of het 'weet' die ouder is dan jij. Zeg in plaats daarvan NIETS, of gebruik direct een parafrase (bijv. 'Dus u voelt zich...?') of checkvraag (bijv. 'Heb ik dat goed begrepen?') om te tonen dat je luistert zonder te pretenderen dat je het al begrijpt.")

    # Additional checks
    if understanding_hits and question_count < max(1, understanding_hits):
        gap_reasons.append("Na het uitspreken van begrip volgden weinig verdiepende of verifiërende vragen.")
    if understanding_hits and paraphrase_hits == 0 and checkvraag_hits == 0:
        gap_reasons.append("Er werd geen parafrase of checkvraag gebruikt om te toetsen of het begrip echt klopte.")
    if coverage < 60 and understanding_hits:
        focus = ", ".join(missing_patterns[:2]) if missing_patterns else "belangrijke patronen"
        gap_reasons.append(f"Je gaf aan dat je het begreep, maar liet nog belangrijke patronen onbesproken ({focus}).")
    if confusion_hits:
        gap_reasons.append("Er zijn signalen van twijfel of verwarring in je antwoorden.")

    gap_detected = len(gap_reasons) > 0
    if gap_detected:
        summary = " ".join(gap_reasons[:3])  # Limit to first 3 reasons
    else:
        summary = "Je parafraseerde of stelde vervolgvragen waardoor je begrip aannemelijk is."

    return {
        "gap_detected": gap_detected,
        "reasons": gap_reasons,
        "exact_phrases": exact_phrases_quoted,
        "gap_issues": gap_issues,
        "summary": summary,
        "student_messages": len(student_messages),
        "understanding_statements": understanding_hits,
        "paraphrase_attempts": paraphrase_hits,
        "checkvraag_attempts": checkvraag_hits,
        "followup_questions": question_count,
        "confusion_signals": confusion_hits,
    }


def normalize_pause_distribution(raw_distribution: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Ensure pause distribution is always expressed as percentages.
    """
    default = {"short": 0.0, "medium": 0.0, "long": 0.0}
    if not raw_distribution:
        return default

    distances = {k: max(0.0, float(raw_distribution.get(k, 0.0))) for k in default}
    total = sum(distances.values())
    if total <= 0:
        return default

    # If totals look like ratios (< 1.5) convert to percentages.
    scale = 100.0 if total <= 1.5 else 1.0
    return {k: round(v * scale if scale == 100.0 else v, 1) for k, v in distances.items()}


def infer_emotion(existing_value: Optional[str], metrics: Dict[str, float]) -> str:
    """
    Determine the perceived emotional tone using provided metadata or heuristics.
    """
    if existing_value:
        return existing_value.lower()

    tempo_variation = metrics.get("tempo_variation", 0)
    filler_ratio = metrics.get("filler_ratio", 0)
    short_pauses = normalize_pause_distribution(metrics.get("pause_distribution")).get("short", 0)

    if tempo_variation > THRESHOLDS["tempo_variation"]["high"] or filler_ratio > THRESHOLDS["filler_ratio"]["medium"]:
        return "stressed"
    if short_pauses > THRESHOLDS["pause_short_ratio"]:
        return "uncertain"
    if tempo_variation < THRESHOLDS["tempo_variation"]["stable"] and filler_ratio < THRESHOLDS["filler_ratio"]["low"]:
        return "calm"
    return "neutral"


def compute_prosody_score(metrics: Dict[str, float], emotion: str) -> float:
    """
    Combine tempo, pause, and tone to approximate a prosody score.
    """
    if "prosody_score" in metrics and metrics["prosody_score"] is not None:
        return float(metrics["prosody_score"])

    base = 78.0
    tempo_penalty = max(0.0, metrics.get("tempo_variation", 0) - THRESHOLDS["tempo_variation"]["stable"]) * 1.2
    filler_penalty = metrics.get("filler_ratio", 0) * 0.6
    pause_avg = metrics.get("average_pause_length", 0)
    pause_penalty = 5.0 if pause_avg < THRESHOLDS["pause_avg"]["short"] else 0.0
    if pause_avg > THRESHOLDS["pause_avg"]["long"]:
        pause_penalty += 6.0

    emotion_adjustment = {"calm": 4.0, "empathetic": 6.0, "neutral": 0.0, "uncertain": -4.0, "stressed": -6.0, "confused": -5.0}
    score = base - tempo_penalty - filler_penalty - pause_penalty + emotion_adjustment.get(emotion, 0.0)
    return max(0.0, min(100.0, round(score, 1)))


def build_feedback_metadata(gordon_result: Optional[Dict], speech_result: Optional[Dict]) -> Dict[str, Any]:
    """
    Compile reusable metadata that can power prompt generation and UI rendering.
    """
    gordon_result = gordon_result or {}
    speech_result = speech_result or {}
    metrics = speech_result.get("metrics", {}) or {}
    confidence = speech_result.get("confidence", {}) or {}
    pause_distribution = normalize_pause_distribution(metrics.get("pause_distribution"))
    
    # Handle field name inconsistency: speech_analysis returns "avg_pause" but we need "average_pause_length"
    if "avg_pause" in metrics and "average_pause_length" not in metrics:
        metrics["average_pause_length"] = metrics["avg_pause"]

    pattern_details = gordon_result.get("pattern_details", {}) or {}

    def _resolve_pattern_label(pattern_entry: Any) -> str:
        """Convert pattern IDs to human-readable names."""
        if isinstance(pattern_entry, dict):
            name = pattern_entry.get("name")
            if name:
                return str(name)
        if isinstance(pattern_entry, int):
            detail = pattern_details.get(str(pattern_entry))
            if detail and detail.get("name"):
                return str(detail["name"])
            return str(pattern_entry)
        if isinstance(pattern_entry, str):
            # Some payloads send IDs as strings, e.g. "3"
            if pattern_entry.isdigit():
                detail = pattern_details.get(pattern_entry)
                if detail and detail.get("name"):
                    return str(detail["name"])
            return pattern_entry
        return str(pattern_entry)

    mentioned_patterns_raw = gordon_result.get("mentioned_patterns", []) or []
    mentioned_pattern_labels = [_resolve_pattern_label(p) for p in mentioned_patterns_raw]

    missing_patterns_raw = gordon_result.get("missing_patterns", []) or []
    missing_pattern_labels = [_resolve_pattern_label(p) for p in missing_patterns_raw]

    metadata = {
        "speech_rate_wpm": round(metrics.get("speech_rate_wpm", 0)),
        "pause_avg": round(metrics.get("average_pause_length", 0.0), 2),
        "pause_distribution": pause_distribution,
        "tempo_variation": round(metrics.get("tempo_variation", 0.0), 1),
        "hesitation_markers": int(metrics.get("hesitation_markers", 0)),
        "volume_stability": metrics.get("volume_stability"),
        "filler_ratio": round(metrics.get("filler_ratio", 0.0), 1),
        "total_words": metrics.get("total_words", 0),  # Add for Rule 6
        "confidence_score": confidence.get("score", 0),
        "confidence_level": confidence.get("level", "medium"),
        "confidence_indicators": confidence.get("indicators", []),
        "speech_summary": scrub_text((speech_result or {}).get("summary")),
        "emotion": infer_emotion((speech_result or {}).get("emotion"), metrics),
        "coverage_percentage": round(gordon_result.get("coverage_percentage", 0)),
        "patterns_mentioned": mentioned_pattern_labels,
        "patterns_missing": missing_pattern_labels,
        "covered_patterns": gordon_result.get("covered_patterns", 0),
        "total_patterns": gordon_result.get("total_patterns", 11),
        "gordon_summary": scrub_text(gordon_result.get("summary")),
    }

    metadata["prosody_score"] = compute_prosody_score(metrics, metadata["emotion"])

    if not metadata["patterns_missing"]:
        # Derive missing patterns using Gordon's 11 base items.
        default_patterns = [
            "Gezondheidsbeleving", "Voeding", "Uitscheiding", "Activiteit", "Slaap",
            "Cognitie", "Zelfbeleving", "Rollen", "Seksualiteit", "Stress", "Waarden"
        ]
        mentioned_lower = {p.lower() for p in metadata["patterns_mentioned"]}
        metadata["patterns_missing"] = [p for p in default_patterns if p.lower() not in mentioned_lower]

    metadata["llm_prompt"] = build_llm_prompt(metadata)
    return metadata


def build_llm_prompt(metadata: Dict[str, Any]) -> str:
    """
    Create the exact Dutch prompt template filled with metric values.
    """
    pause_distribution_text = format_pause_distribution_text(metadata["pause_distribution"])
    patterns_mentioned_text = ", ".join(metadata["patterns_mentioned"]) or "geen patronen genoemd"
    patterns_missing_text = ", ".join(metadata["patterns_missing"]) or "geen ontbrekende patronen"

    prompt = f"""
Je bent een professionele beoordelaar van gespreksvaardigheden voor HBO-V studenten.
Gebruik ALTIJD de onderstaande structuur en schrijf in duidelijk, vriendelijk en professioneel Nederlands.
Verwijs naar de METRICS (onderaan) om je feedback concreet en specifiek te maken.
Vermijd vage opmerkingen, wees feitelijk, empathisch en gericht op leerdoelen.

BELANGRIJKE REGELS:
- Als dekking < 27% (minder dan 3/11 patronen): DE toon MOET kritisch zijn. Geef maximaal 1 compliment en focus op wat ontbreekt.
- Quote ALTIJD exacte studentuitdrukkingen (bijv. "Je zei: 'ik snap het'...").
- Noem exacte fillers als die voorkomen (bijv. "Je gebruikte 'eh' meerdere keren").
- Wees realistisch: bij lage dekking of lage prosodie/vertrouwen moet de toon kritischer zijn.

=====================================
### 1. Complimenten
Noem 2–3 positieve observaties over:
- tempo, rust, helderheid, empathie
- Gordon patronen die goed gestart zijn
- professionele houding
Gebruik minstens één concrete waarde uit de metrics (bijv. {metadata['speech_rate_wpm']} wpm, pauzelengte, emotionele toon).

### 2. Communicatiegedrag
Analyseer:
- spreektempo (speech_rate_wpm, tempo_variatie)
- pauzes (pause_avg, pause_distribution)
- fillers (filler_ratio)
- emotionele toon (emotion)
- prosodie (prosody_score)

Leg uit wat deze patronen zeggen over zekerheid, empathie en gesprekstechniek.

### 3. Gordon-patronen
Bespreek:
- welke patronen zijn genoemd ({patterns_mentioned_text})
- welke patronen ontbreken ({patterns_missing_text})
- waarom deze relevant zijn
- welke vervolgvragen hierbij passen

### 4. Klinische redenering
Geef 2–3 gerichte adviezen over:
- gespreksopbouw
- prioriteiten stellen
- omgaan met emoties van de patiënt
- anamnesetechnieken

### 5. Concrete vervolgstappen
Geef 3–5 korte, haalbare acties die de student direct kan toepassen bij een volgende oefening.

=====================================
METRICS
Spreeksnelheid: {metadata['speech_rate_wpm']} wpm
Tempo variatie: {metadata['tempo_variation']}
Gem. pauzelengte: {metadata['pause_avg']}
Pauze verdeling: {pause_distribution_text}
Filler ratio: {metadata['filler_ratio']}
Emotie: {metadata['emotion']}
Prosodie: {metadata['prosody_score']}
Gordon dekking: {metadata['coverage_percentage']}%
Patronen genoemd: {patterns_mentioned_text}
Patronen gemist: {patterns_missing_text}
""".strip()

    return prompt


def generate_llm_section_default(header: str, metadata: Dict[str, Any]) -> str:
    """
    Ensure every section has at least a short, metric-driven fallback.
    """
    pause_distribution = metadata["pause_distribution"]
    pause_info = f"{pause_distribution['short']}% kort / {pause_distribution['medium']}% middel / {pause_distribution['long']}% lang"

    if header == "Complimenten":
        return (
            f"- Stevige basis: {metadata['covered_patterns']}/{metadata['total_patterns']} patronen benoemd "
            f"({metadata['coverage_percentage']}%).\n"
            f"- Spreeksnelheid {metadata['speech_rate_wpm']} wpm met prosodie {metadata['prosody_score']}/100."
        )
    if header == "Communicatiegedrag":
        return (
            f"- Tempo-variatie {metadata['tempo_variation']}% met gemiddelde pauze {metadata['pause_avg']}s "
            f"({pause_info}).\n"
            f"- Tonaliteit: {metadata['emotion']} | Zelfvertrouwen {metadata['confidence_score']}/100."
        )
    if header == "Gordon-patronen":
        missing = ", ".join(metadata["patterns_missing"][:3]) or "geen clear gaps"
        mentioned = ", ".join(metadata["patterns_mentioned"][:3]) or "n.v.t."
        return (
            f"- Genoemd: {mentioned}.\n"
            f"- Mist nog: {missing}. Richt je op de ontbrekende patronen bij de volgende anamnese."
        )
    if header == "Klinische redenering":
        return (
            "- Hou de anamnesestroom logisch: koppel observaties direct aan vervolgvragen.\n"
            f"- Gebruik de ontbrekende patronen ({', '.join(metadata['patterns_missing'][:2])}) als kapstok."
        )
    if header == "Concrete vervolgstappen":
        return (
            "- Plan 1 oefensessie focussen op rustiger pauzes (< "
            f"{THRESHOLDS['pause_avg']['long']}s) en minder hesitaties ({metadata['hesitation_markers']} gemeten).\n"
            "- Bereid 2 nieuwe vragen voor rond de ontbrekende patronen."
        )
    return ""


def sanitize_llm_output(conversation_feedback, metadata: Dict[str, Any]) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Clean LLM feedback, enforce required headers, and attach metric-driven fallbacks.
    """
    lecturer_notes = None
    raw_text = ""
    if isinstance(conversation_feedback, dict):
        raw_text = conversation_feedback.get("text") or conversation_feedback.get("content") or ""
        lecturer_notes = conversation_feedback.get("lecturer_notes")
    else:
        raw_text = conversation_feedback or ""

    sanitized_text = scrub_text(raw_text)
    sections = {title: "" for title in LLM_SECTION_TITLES}
    current = LLM_SECTION_TITLES[0]
    buffer: List[str] = []

    if sanitized_text:
        pattern_map = {title.lower(): title for title in LLM_SECTION_TITLES}
        for line in sanitized_text.splitlines():
            stripped = line.strip().strip(":").lower()
            matched = next((pattern_map[name] for name in pattern_map if stripped.startswith(name)), None)
            if matched:
                sections[current] = "\n".join(buffer).strip()
                current = matched
                buffer = []
                continue
            buffer.append(line)
        sections[current] = "\n".join(buffer).strip()

    for header in LLM_SECTION_TITLES:
        if not sections[header]:
            sections[header] = generate_llm_section_default(header, metadata)
        elif header == "Complimenten":
            # Clamp complimenten if coverage or prosody are weak.
            if metadata["coverage_percentage"] < 40 or metadata["prosody_score"] < THRESHOLDS["prosody"]["ok"]:
                compliments = sections[header].split("\n")
                sections[header] = "\n".join(compliments[:2]).strip()

    return sections, scrub_text(lecturer_notes)


def count_sentences(text: Optional[str]) -> int:
    """Count sentences in text (simple heuristic based on sentence-ending punctuation)."""
    if not text:
        return 0
    # Count sentences by splitting on sentence-ending punctuation
    import re
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def build_summary_section(
    metadata: Dict[str, Any],
    conversation_history: Optional[str] = None,
    conversation_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create the top summary with badges and quick metrics.
    Rule 4: If coverage < 3/11 → require strong negative feedback (orange/red).
    Rule 5: If history < 5 sentences → force warning.
    """
    lines = ["=== 1. Samenvatting ==="]
    analysis_block = conversation_analysis or metadata.get("conversation_analysis_block") or {}
    analysis_flags = analysis_block.get("flags", {})
    analysis_examples = analysis_block.get("extracted_examples", {})
    exact_phrases = analysis_block.get("exact_phrases_used", {})

    # Overall verdict based on coverage, prosody, and comprehension check.
    coverage = metadata["coverage_percentage"]
    covered_patterns = metadata.get("covered_patterns", 0)
    prosody = metadata["prosody_score"]
    gap_result = metadata.get("understanding_gap") or {}
    has_gap = gap_result.get("gap_detected")

    # Rule 4: Low coverage handling (< 3/11 patterns = ~27%)
    is_low_coverage = covered_patterns < 3 or coverage < 27.3
    
    # Rule 5: Check if history is very short
    sentence_count = count_sentences(conversation_history) if conversation_history else 0
    is_very_short = sentence_count < 5

    # Determine overall level - Rule 4: Force orange/red for low coverage
    if is_low_coverage:
        overall_level = "low"
        missing_patterns = metadata.get("patterns_missing", [])
        missing_list = ", ".join(missing_patterns[:3]) if missing_patterns else "belangrijke patronen"
        overall_reason = f"Door slechts {covered_patterns} patronen te behandelen, mis je belangrijke informatie die essentieel is voor een veilige anamnese. Ontbrekende patronen: {missing_list}."
    elif coverage >= 70 and prosody >= THRESHOLDS["prosody"]["good"] and not has_gap:
        overall_level = "high"
        overall_reason = "Goede dekking, sterke prosodie en geen duidelijke begripskloof."
    elif coverage >= 40 and prosody >= THRESHOLDS["prosody"]["ok"]:
        overall_level = "medium"
        overall_reason = "Redelijke dekking of prosodie, maar er is ruimte voor verdieping of scherpere opvolging."
    else:
        overall_level = "low" if coverage < 40 or prosody < THRESHOLDS["prosody"]["ok"] else "medium"
        reason_parts = []
        if coverage < 40:
            reason_parts.append(f"lage dekking ({coverage}%)")
        if prosody < THRESHOLDS["prosody"]["ok"]:
            reason_parts.append(f"prosodie {prosody}/100")
        if has_gap or analysis_flags.get("comprehension_gap"):
            reason_parts.append("begrip niet overtuigend getoond")
        overall_reason = "; ".join(reason_parts) if reason_parts else "verbeterpunten vereist."

    # Rule 8: Use emoji icons
    icon = "🟢" if overall_level == "high" else ("🟠" if overall_level == "medium" else "🔴")
    lines.append(f"- {icon} Beoordeling: {overall_reason}")
    
    # Rule 5: Add warning for very short history
    if is_very_short:
        lines.append(f"- ⚠️ Het gesprek was zeer kort ({sentence_count} zinnen), waardoor onvoldoende inzicht ontstond in de situatie van de patiënt.")
    confidence_score = metadata["confidence_score"]
    confidence_level = metadata["confidence_level"]

    if confidence_level == "high" or confidence_score >= THRESHOLDS["confidence"]["high"]:
        lines.append(f"- ✅ Zelfvertrouwen: {confidence_score}/100 – je klonk zeker en rustig.")
    elif confidence_level == "medium":
        lines.append(f"- ⚠️ Zelfvertrouwen: {confidence_score}/100 – nog winst te behalen in rust en overtuiging.")
    else:
        lines.append(f"- ❌ Zelfvertrouwen: {confidence_score}/100 – oefen met ademhaling en eye contact.")

    filler_ratio = metadata["filler_ratio"]
    filler_tokens = exact_phrases.get("filler_words", [])
    filler_summary = summarize_filler_tokens(filler_tokens)
    if analysis_flags.get("filler_issue") and filler_summary:
        lines.append(f"- ❌ Stopwoorden: {filler_summary} hoorde ik letterlijk in je transcript.")
    elif filler_ratio <= THRESHOLDS["filler_ratio"]["low"]:
        lines.append("- ✅ Stopwoorden: vrijwel geen fillers gebruikt.")
    elif filler_ratio <= THRESHOLDS["filler_ratio"]["medium"]:
        lines.append(f"- ⚠️ Stopwoorden: {filler_ratio}% – let op overmatig 'euh' of herhalingen.")
    else:
        lines.append(f"- ❌ Stopwoorden: {filler_ratio}% – vertraag je tempo om fillers te beperken.")

    coverage = metadata["coverage_percentage"]
    if coverage >= 70:
        lines.append(f"- ✅ Gordon patronen: {metadata['covered_patterns']}/{metadata['total_patterns']} ({coverage}%) behandeld.")
    elif coverage >= 40:
        lines.append(f"- ⚠️ Gordon patronen: {metadata['covered_patterns']}/{metadata['total_patterns']} ({coverage}%) – pak meer domeinen mee.")
    else:
        lines.append(f"- ❌ Gordon patronen: slechts {coverage}% dekking, plan bewuste vervolgvragen.")
        if analysis_flags.get("low_coverage_issue"):
            lines.append(f"- ❌ You covered only {covered_patterns} out of 11 patterns; vul de ontbrekende patronen in met gerichte vragen.")

    gap_result = metadata.get("understanding_gap") or {}
    understanding_examples = analysis_examples.get("understanding_examples") or []

    if gap_result.get("student_messages", 0) == 0 and not understanding_examples:
        lines.append("- ℹ️ Begripscontrole: geen uitspraken om begrip te toetsen.")
    elif gap_result.get("gap_detected") or analysis_flags.get("comprehension_gap"):
        # Rule 3: Quote exact phrases and give explicit directive
        gap_examples = understanding_examples or gap_result.get("exact_phrases", [])
        if gap_examples:
            phrases_quoted = "', '".join(set(gap_examples[:3]))
            lines.append(f"- ❌ Begripscontrole: '{phrases_quoted}' zonder parafrase binnen 2 zinnen; 'ik begrijp het' is onvoldoende. Vat samen wat u hoort of stel een checkvraag.")
        else:
            lines.append(f"- ❌ Begripscontrole: {gap_result.get('summary', 'Begrip niet overtuigend getoond.')}")
    else:
        lines.append("- ✅ Begripscontrole: parafrases en vervolgvragen maakten je begrip overtuigend.")

    # Always surface the exact begrip-claims so the student sees what was said
    if understanding_examples:
        phrases_quoted = "', '".join(set(understanding_examples[:3]))
        lines.append(f"- Gehoorde begrip-zinnen: '{phrases_quoted}'. Gebruik ze alleen met directe parafrase/checkvraag.")

    # Rule 4: List missing patterns for low coverage
    missing_patterns = metadata.get("patterns_missing") or []
    if is_low_coverage and missing_patterns:
        top_missing = ", ".join(missing_patterns[:4])
        lines.append(f"- ❌ Ontbrekende patronen: {top_missing} (essentieel voor veilige anamnese)")
    elif missing_patterns:
        top_missing = ", ".join(missing_patterns[:2])
        lines.append(f"- Volgende focus: vraag door op {top_missing} met concrete voorbeelden.")

    # Abrupt ending check
    if analysis_flags.get("abrupt_ending_issue") and analysis_examples.get("abrubt_ending_examples"):
        lines.append(f"- ❌ Afsluiting: '{analysis_examples['abrubt_ending_examples'][0]}' klonk abrupt en kwam zonder samenvatting.")

    lines.append(
        f"- Metrics: {metadata['speech_rate_wpm']} wpm | tempo-variatie {metadata['tempo_variation']}% | "
        f"pauze {metadata['pause_avg']}s | prosodie {metadata['prosody_score']}/100 | emotie {metadata['emotion']}."
    )
    return "\n".join(lines)


def format_pause_distribution_text(pause_distribution: Dict[str, float]) -> str:
    return f"{pause_distribution['short']}% kort / {pause_distribution['medium']}% middel / {pause_distribution['long']}% lang"


def build_speech_section(
    speech_result: Optional[Dict],
    metadata: Dict[str, Any],
    conversation_analysis: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Detailed speech analytics with numeric metrics.
    Rule 2: MUST include filler detection and density per 10 words.
    Rule 6: Include speech rate feedback.
    """
    analysis_block = conversation_analysis or metadata.get("conversation_analysis_block") or {}
    if not speech_result and not analysis_block:
        return None

    pause_text = format_pause_distribution_text(metadata["pause_distribution"])
    lines = ["=== 3. Spraak Analyse ===", "**Belangrijkste metingen**"]
    lines.append(f"- Spreeksnelheid: {metadata['speech_rate_wpm']} wpm")
    
    # Rule 6: Speech rate feedback
    speech_rate = metadata['speech_rate_wpm']
    if speech_rate > 150:
        lines.append("  → Je sprak vrij snel; iets meer rust helpt de patiënt zich gehoord te voelen.")
    elif speech_rate < 100 and speech_rate > 0:
        lines.append("  → Je sprak erg langzaam, wat onzeker kan overkomen.")
    
    lines.append(f"- Tempo-variatie: {metadata['tempo_variation']}%")
    lines.append(f"- Pauzedistributie: {pause_text}")
    lines.append(f"- Gemiddelde pauzeduur: {metadata['pause_avg']} s")
    
    # Rule 2: Filler/hesitation detection - MANDATORY if any fillers found
    metrics = (speech_result or {}).get("metrics", {}) if speech_result else {}
    filler_tokens = (analysis_block.get("exact_phrases_used") or {}).get("filler_words", [])
    filler_count = metadata.get("hesitation_markers", 0) or len(filler_tokens)
    filler_ratio = metadata.get('filler_ratio', 0)
    total_words = metadata.get('total_words', 0) or metrics.get('total_words', 0) or analysis_block.get("summary_findings", {}).get("conversation_length", 0)

    if filler_count > 0:
        # Calculate density per 10 words
        filler_density = (filler_count / total_words * 10) if total_words > 0 else 0
        lines.append(f"- Opvulgeluidjes/fillers: {filler_count} keer gebruikt")
        lines.append(f"- Filler-dichtheid: {filler_density:.1f} per 10 woorden")
        lines.append("  → Opvulgeluidjes verminderen de duidelijkheid van je communicatie.")
        
        # Rule 2C: Severity indicator if > 3
        if filler_count > 3:
            lines.append(f"  → Je gebruikte veel opvulgeluidjes ({filler_count} keer), wat de professionaliteit verlaagt.")
    elif filler_tokens:
        filler_summary = summarize_filler_tokens(filler_tokens)
        lines.append(f"- Opvulgeluidjes/fillers: transcript toont {filler_summary}")
    else:
        lines.append("- Opvulgeluidjes/fillers: geen gedetecteerd")

    if filler_tokens:
        filler_summary = summarize_filler_tokens(filler_tokens)
        filler_examples = (analysis_block.get("extracted_examples") or {}).get("filler_examples", [])
        lines.append(f"- Exacte fillers in transcript: {filler_summary}")
        if filler_examples:
            lines.append(f"  → Gehoord in: \"{filler_examples[0]}\"")
    
    lines.append(f"- Volume-stabiliteit: {metadata['volume_stability'] or 'n.v.t.'}")
    lines.append(f"- Prosodie: {metadata['prosody_score']}/100")
    lines.append(f"- Gevoelstoon: {metadata['emotion']}")
    
    # Rule 6: Total words feedback
    if total_words < 30:
        lines.append(f"\n⚠️ Door het beperkte aantal woorden ({total_words}) kreeg de patiënt weinig ruimte.")

    summary = metadata["speech_summary"]
    if summary:
        lines.append("\n**Interpretatie**")
        lines.append(summary)

    indicators = metadata.get("confidence_indicators") or []
    if indicators:
        lines.append("\n**Indicatoren zelfvertrouwen**")
        lines.extend([f"- {indicator}" for indicator in indicators[:5]])

    return "\n".join(lines)


def build_conversation_skills_section(
    llm_sections: Dict[str, str],
    metadata: Dict[str, Any],
    conversation_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Section 2: Gespreksvaardigheden – grounded in transcript plus LLM observations.
    """
    analysis_block = conversation_analysis or metadata.get("conversation_analysis_block") or {}
    exact = analysis_block.get("exact_phrases_used", {})
    flags = analysis_block.get("flags", {})
    examples = analysis_block.get("extracted_examples", {})

    lines = ["=== 2. Gespreksvaardigheden ==="]

    open_qs = exact.get("open_questions", [])
    closed_qs = exact.get("closed_questions", [])
    paraphrases = exact.get("paraphrases", [])
    empathy_attempts = exact.get("empathy_attempts", [])
    filler_words = summarize_filler_tokens(exact.get("filler_words", []))

    if open_qs or closed_qs:
        lines.append(f"- Vragen: {len(open_qs)} open, {len(closed_qs)} gesloten.")
        if open_qs:
            lines.append(f"  → Voorbeeld open vraag: \"{open_qs[0]}\"")
        if closed_qs:
            lines.append(f"  → Voorbeeld gesloten vraag: \"{closed_qs[0]}\"")
    if paraphrases:
        lines.append(f"- Parafrasepogingen: {len(paraphrases)} (bv. \"{paraphrases[0]}\")")
    else:
        lines.append("- ❌ Geen parafrase gehoord; vat binnen 2 zinnen samen wat u hoorde.")
    if empathy_attempts:
        lines.append(f"- Empathie: {len(empathy_attempts)} keer, zoals \"{empathy_attempts[0]}\"")
    if filler_words:
        lines.append(f"- Fillers in gesprek: {filler_words}")
    if flags.get("abrupt_ending_issue") and examples.get("abrubt_ending_examples"):
        lines.append(f"- ❌ Abrupte afsluiting: \"{examples['abrubt_ending_examples'][0]}\" zonder samenvatting.")

    if llm_sections:
        lines.append("\n**LLM-observaties**")
        for header in ("Complimenten", "Communicatiegedrag", "Klinische redenering"):
            content = (llm_sections.get(header) or "").strip()
            if content:
                lines.append(f"{header}:\n{content}")

    return "\n".join(lines)


def build_understanding_section(
    gap_result: Dict[str, Any],
    conversation_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Summarize gaps between uitgesproken en aangetoond begrip.
    Rule 1: Must quote exact phrases that caused issues.
    Rule 3: Always quote exact student phrases.
    """
    lines = ["=== 4. Begripstoetsing ==="]
    analysis_block = conversation_analysis or {}
    analysis_flags = analysis_block.get("flags", {})
    analysis_examples = analysis_block.get("extracted_examples", {})
    analysis_phrases = analysis_block.get("exact_phrases_used", {})
    understanding_examples = analysis_examples.get("understanding_examples") or []

    if (not gap_result or gap_result.get("student_messages", 0) == 0) and not analysis_examples.get("understanding_examples"):
        lines.append("- Geen studentuitspraken beschikbaar om begrip te toetsen.")
        return "\n".join(lines)

    lines.append(
        f"- Uitgesproken begrip: {gap_result.get('understanding_statements', 0)} | parafrases: "
        f"{gap_result.get('paraphrase_attempts', 0)} | checkvragen: {gap_result.get('checkvraag_attempts', 0)} | vervolgvragen: {gap_result.get('followup_questions', 0)}."
    )

    if analysis_flags.get("comprehension_gap"):
        unique_examples = []
        for ex in understanding_examples or analysis_phrases.get("understanding_phrases", []):
            if ex not in unique_examples:
                unique_examples.append(ex)
        if unique_examples:
            quoted = "', '".join(unique_examples[:3])
            lines.append(f"- ❌ Begrip claimen zonder check: '{quoted}'. 'Ik begrijp het' is onvoldoende; parafraseer binnen 2 zinnen of stel een checkvraag.")
        else:
            lines.append("- ❌ Begrip claimen zonder check: parafrase of checkvraag ontbrak binnen 2 zinnen na je begrip-uitspraak.")
    elif gap_result.get("gap_detected"):
        # Rule 1B & Rule 3: Quote exact phrases
        exact_phrases = gap_result.get("exact_phrases", [])
        gap_issues = gap_result.get("gap_issues", [])
        
        if exact_phrases:
            # Group phrases by type and count
            phrase_counts = {}
            for phrase in exact_phrases:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            
            # Get all unique phrases for a comprehensive feedback
            all_phrases = list(phrase_counts.keys())
            phrases_text = "', '".join(all_phrases[:4])  # Show up to 4 examples
            total_count = sum(phrase_counts.values())
            
            if total_count == 1:
                lines.append(f"- ❌ VERMIJD: Je zei '{phrases_text}' tegen iemand die ouder is dan jij. Dit is onbeleefd en onprofessioneel - je kunt niet beweren dat je iemand 'begrijpt' of 'snapt' die ouder is dan jij. Zeg in plaats daarvan NIETS, of gebruik direct een parafrase ('Dus u zegt dat...?') of checkvraag ('Heb ik dat goed begrepen?') om te tonen dat je luistert zonder te pretenderen dat je het al begrijpt.")
            else:
                lines.append(f"- ❌ VERMIJD: Je gebruikte {total_count} keer uitspraken zoals '{phrases_text}' tegen iemand die ouder is dan jij. Dit is onbeleefd en onprofessioneel - je kunt niet beweren dat je iemand 'begrijpt', 'snapt', het 'weet' of dat het 'klopt' die ouder is dan jij. Zeg in plaats daarvan NIETS, of gebruik direct een parafrase ('Dus u voelt zich...?') of checkvraag ('Klopt het dat...?') om te tonen dat je luistert zonder te pretenderen dat je het al begrijpt.")
        
        # Add other reasons
        reasons = gap_result.get("reasons", [])
        for reason in reasons:
            if not any(phrase in reason for phrase in exact_phrases):  # Avoid duplicates
                lines.append(f"- {reason}")
    else:
        lines.append("- ✅ Geen duidelijke kloof tussen gedacht en getoond begrip; je parafraseerde of stelde checkvragen waardoor je begrip aannemelijk is.")

    return "\n".join(lines)


def build_gordon_section(metadata: Dict[str, Any]) -> str:
    """
    Describe Gordon pattern coverage and missing aspects.
    Rule 4: If coverage < 3/11, explicitly list missing patterns and explain why they're essential.
    """
    lines = ["=== 5. Gordon-patronen Analyse ==="]
    coverage = metadata["coverage_percentage"]
    covered_patterns = metadata['covered_patterns']
    total_patterns = metadata['total_patterns']
    
    lines.append(f"- Dekking: {covered_patterns}/{total_patterns} ({coverage:.1f}%)")

    # Rule 4: Strong feedback for low coverage
    is_low_coverage = covered_patterns < 3 or coverage < 27.3
    if is_low_coverage:
        lines.append(f"- ⚠️ Lage dekking: Door slechts {covered_patterns} patronen te behandelen, mis je belangrijke informatie die essentieel is voor een veilige anamnese.")
        lines.append(f"- ❌ You covered only {covered_patterns} out of 11 patterns; zorg voor een bredere anamnese.")

    mentioned = ", ".join(metadata["patterns_mentioned"]) if metadata["patterns_mentioned"] else "geen vermeld"
    missing = ", ".join(metadata["patterns_missing"]) if metadata["patterns_missing"] else "n.v.t."
    lines.append(f"- Genoemde patronen: {mentioned}")
    
    # Rule 4: Explicitly list missing patterns for low coverage
    if is_low_coverage and metadata["patterns_missing"]:
        lines.append(f"- ❌ Ontbrekende patronen (essentieel): {missing}")
    else:
        lines.append(f"- Ontbrekende patronen: {missing}")

    if metadata["gordon_summary"]:
        lines.append(f"- Samenvatting: {metadata['gordon_summary']}")

    top_missing = ", ".join(metadata["patterns_missing"][:3]) if metadata["patterns_missing"] else "n.v.t."
    if top_missing != "n.v.t.":
        lines.append(f"- Focus voor volgende keer: {top_missing}")

    # Add concrete follow-up questions for the most relevant missing patterns.
    follow_ups = []
    for pattern in metadata["patterns_missing"][:2]:
        question = PATTERN_QUESTIONS.get(pattern) or PATTERN_QUESTIONS.get(pattern.split(" / ")[0])
        if question:
            follow_ups.append(f"Stel: \"{question}\"")
    if follow_ups:
        lines.append(f"- Vervolgvragen: {' | '.join(follow_ups)}")
    return "\n".join(lines)


def build_action_items(metadata: Dict[str, Any]) -> str:
    """
    Use speech + Gordon metrics to craft concrete action steps.
    Rule 7: MUST include 2 strengths, 3 improvements, 1 specific communication technique.
    Rule 2: Include filler reduction action if fillers > 3.
    """
    analysis_block = metadata.get("conversation_analysis_block") or {}
    analysis_flags = analysis_block.get("flags", {})
    analysis_exact = analysis_block.get("exact_phrases_used", {})
    analysis_examples = analysis_block.get("extracted_examples", {})

    strengths: List[str] = []
    improvements: List[str] = []
    techniques: List[str] = []
    
    filler_ratio = metadata["filler_ratio"]
    filler_count = metadata.get("hesitation_markers", 0)
    analysis_filler_count = len(analysis_exact.get("filler_words", []))
    if filler_count == 0 and analysis_filler_count:
        filler_count = analysis_filler_count
    tempo_variation = metadata["tempo_variation"]
    speech_rate = metadata["speech_rate_wpm"]
    pause_distribution = metadata["pause_distribution"]
    pause_avg = metadata["pause_avg"]
    emotion = metadata["emotion"]
    hesitations = metadata["hesitation_markers"]
    prosody = metadata["prosody_score"]
    volume_stability = metadata["volume_stability"]
    gap_result = metadata.get("understanding_gap") or {}
    coverage = metadata["coverage_percentage"]
    
    # Collect strengths
    if prosody >= THRESHOLDS["prosody"]["good"]:
        strengths.append("Je prosodie was sterk; je klonk natuurlijk en betrokken.")
    if filler_ratio <= THRESHOLDS["filler_ratio"]["low"]:
        strengths.append("Je gebruikte weinig opvulgeluidjes, wat duidelijkheid bevordert.")
    if THRESHOLDS["speech_rate"]["slow"] <= speech_rate <= THRESHOLDS["speech_rate"]["fast"]:
        strengths.append("Je spreektempo was goed gebalanceerd en aangepast aan de situatie.")
    if pause_avg >= THRESHOLDS["pause_avg"]["short"] and pause_avg <= THRESHOLDS["pause_avg"]["long"]:
        strengths.append("Je pauzes waren goed getimed en gaven ruimte voor reactie.")
    if coverage >= 60:
        strengths.append(f"Je behandelde een breed scala aan patronen ({coverage:.0f}% dekking).")
    if not gap_result.get("gap_detected"):
        strengths.append("Je toetste je begrip door parafrases of checkvragen te stellen.")
    
    # Collect improvements
    if tempo_variation > THRESHOLDS["tempo_variation"]["high"]:
        improvements.append("Probeer rustiger en consistenter te spreken; stabiliseer je tempo per vraag.")
    if speech_rate < THRESHOLDS["speech_rate"]["slow"]:
        improvements.append("Verhoog je spreeksnelheid licht zodat het gesprek levendiger blijft (doel 105-125 wpm).")
    elif speech_rate > THRESHOLDS["speech_rate"]["fast"]:
        improvements.append("Vertraag bewust door na elke vraag een korte pauze te nemen voor helderheid.")
    if pause_distribution["short"] > THRESHOLDS["pause_short_ratio"] or pause_avg < THRESHOLDS["pause_avg"]["short"]:
        improvements.append("Kort en veel pauzes kunnen onzekerheid tonen; adem diep uit voordat je reageert.")
    # Rule 2B: Filler reduction action
    if filler_count > 3:
        improvements.append("Verminder het aantal fillers zoals 'eh', omdat dit onzeker overkomt. Gebruik korte pauzes in plaats van opvulgeluidjes.")
    elif filler_ratio > THRESHOLDS["filler_ratio"]["medium"]:
        improvements.append("Verminder opvulgeluidjes door steekwoorden vooraf te noteren en rustiger te spreken.")
    elif analysis_flags.get("filler_issue") and analysis_exact.get("filler_words"):
        filler_summary = summarize_filler_tokens(analysis_exact.get("filler_words", []))
        improvements.append(f"Schrap de fillers {filler_summary} uit je vragen; neem een korte stilte in plaats van 'eh'.")
    if emotion in {"uncertain", "stressed"}:
        improvements.append("Je klonk wat onzeker; vertraag je ademhaling en vat antwoorden samen om vertrouwen te tonen.")
    if prosody < THRESHOLDS["prosody"]["ok"]:
        improvements.append("Werk aan vocale variatie door sleutelwoorden te benadrukken en toonhoogte licht te variëren.")
    if volume_stability and volume_stability < THRESHOLDS["volume_stability"]:
        improvements.append("Houd je volume stabiel door rechtop te zitten en uit te ademen tijdens het spreken.")
    if coverage < 60:
        missing = ", ".join(metadata["patterns_missing"][:3])
        improvements.append(f"Plan vragen rond ontbrekende patronen ({missing}) om vollediger te screenen.")
    if analysis_flags.get("abrupt_ending_issue") and analysis_examples.get("abrubt_ending_examples"):
        improvements.append(f"Sluit niet af met \"{analysis_examples['abrubt_ending_examples'][0]}\"; eindig met een korte samenvatting en een bedankje.")
    if gap_result.get("gap_detected"):
        exact_phrases = gap_result.get("exact_phrases", [])
        if exact_phrases:
            phrases_example = "', '".join(exact_phrases[:3])
            improvements.append(f"VERMIJD uitspraken zoals '{phrases_example}' tegen iemand die ouder is - dit is onbeleefd omdat je niet kunt beweren dat je iemand 'begrijpt' die ouder is. In plaats daarvan: zeg NIETS, of gebruik direct een parafrase ('Dus u zegt dat...?') of checkvraag ('Heb ik dat goed begrepen?').")
        else:
            improvements.append("VERMIJD uitspraken zoals 'ik begrijp het', 'ja ik snap het', 'i understand', 'i get it' tegen iemand die ouder is - dit is onbeleefd omdat je niet kunt beweren dat je iemand 'begrijpt' die ouder is. In plaats daarvan: zeg NIETS, of gebruik direct een parafrase ('Dus u zegt dat...?') of checkvraag ('Heb ik dat goed begrepen?').")
    elif analysis_flags.get("comprehension_gap"):
        understanding_examples = analysis_examples.get("understanding_examples") or analysis_exact.get("understanding_phrases", [])
        if understanding_examples:
            phrases_example = "', '".join(understanding_examples[:3])
            improvements.append(f"VERMIJD uitspraken zoals '{phrases_example}' zonder parafrase; vat binnen 2 zinnen samen of stel een checkvraag.")
        else:
            improvements.append("VERMIJD uitspraken zoals 'ik begrijp het' zonder directe parafrase of checkvraag; toon begrip door samen te vatten.")

    # Collect specific communication techniques (Rule 7)
    if gap_result.get("gap_detected"):
        exact_phrases = gap_result.get("exact_phrases", [])
        if exact_phrases:
            phrases_example = "', '".join(exact_phrases[:2])
            techniques.append(f"Techniek: VERMIJD uitspraken zoals '{phrases_example}' tegen iemand die ouder is - dit is onbeleefd. In plaats daarvan: zeg NIETS (luister gewoon), of gebruik direct een parafrase ('Dus u voelt zich...?') of checkvraag ('Heb ik dat goed begrepen?') om te tonen dat je luistert zonder te pretenderen dat je het al begrijpt.")
        else:
            techniques.append("Techniek: VERMIJD uitspraken zoals 'ik begrijp het', 'i understand', 'i get it' tegen iemand die ouder is - dit is onbeleefd omdat je niet kunt beweren dat je iemand 'begrijpt' die ouder is. In plaats daarvan: zeg NIETS (luister gewoon), of gebruik direct een parafrase ('Dus u voelt zich...?') of checkvraag ('Heb ik dat goed begrepen?') om te tonen dat je luistert zonder te pretenderen dat je het al begrijpt.")
    elif analysis_flags.get("comprehension_gap"):
        understanding_examples = analysis_examples.get("understanding_examples") or analysis_exact.get("understanding_phrases", [])
        if understanding_examples:
            phrases_example = "', '".join(understanding_examples[:2])
            techniques.append(f"Techniek: gebruik geen '{phrases_example}' zonder parafrase; sluit aan met 'Dus u zegt dat...' of 'Heb ik dat goed begrepen?'.")
        else:
            techniques.append("Techniek: vervang 'ik begrijp het' door een parafrase ('Dus u zegt dat...') gevolgd door een checkvraag.")
    else:
        # Suggest techniques based on what's missing
        if gap_result.get("paraphrase_attempts", 0) == 0:
            techniques.append("Techniek: Oefen met parafraseren - vat kort samen wat de patiënt zei met 'Dus u zegt dat...' of 'Als ik het goed begrijp...'")
        if gap_result.get("checkvraag_attempts", 0) == 0:
            techniques.append("Techniek: Stel checkvragen zoals 'Heb ik dat goed begrepen?' of 'Klopt het dat u...?' om je begrip te toetsen.")
        if not techniques:
            missing_patterns = metadata.get("patterns_missing", [])
            if missing_patterns:
                techniques.append("Techniek: Stel open vragen (begin met 'Hoe', 'Wat', 'Waarom') om dieper in te gaan op ontbrekende patronen.")
            else:
                techniques.append("Techniek: Gebruik empathische reflectie - herhaal de emotie die je hoort: 'Dat klinkt moeilijk voor u' of 'U voelt zich...'")

    # Ensure we have at least 2 strengths, 3 improvements, 1 technique
    if len(strengths) < 2:
        fallback_strengths = [
            "Je startte het gesprek professioneel.",
            "Je luisterde actief naar de patiënt.",
            "Je toonde interesse in de situatie van de patiënt.",
        ]
        for s in fallback_strengths:
            if s not in strengths:
                strengths.append(s)
            if len(strengths) >= 2:
                break
    
    if len(improvements) < 3:
        fallback_improvements = [
            "Gebruik een checklijst met Gordon patronen om structuur te houden.",
            "Herhaal de kernwoorden van de patiënt om empathie te bevestigen.",
            "Noteer tijdens het gesprek kort wat al besproken is om vervolgvragen beter te plannen.",
        ]
        for imp in fallback_improvements:
            if imp not in improvements:
                improvements.append(imp)
            if len(improvements) >= 3:
                break
    
    if not techniques:
        techniques.append("Techniek: Gebruik open vragen die beginnen met 'Hoe', 'Wat' of 'Waarom' om meer informatie te krijgen.")

    lines = ["=== 6. Actiepunten ==="]
    lines.append("**Sterke punten (2):**")
    for strength in strengths[:2]:
        lines.append(f"- ✅ {strength}")
    
    lines.append("\n**Verbeterpunten (3):**")
    for improvement in improvements[:3]:
        lines.append(f"- 🔧 {improvement}")
    
    lines.append("\n**Specifieke communicatietechniek voor volgende keer:**")
    lines.append(f"- 📚 {techniques[0]}")
    
    # Optional: Add one concrete question suggestion if coverage is low
    if coverage < 60 and metadata.get("patterns_missing"):
        pattern = metadata["patterns_missing"][0]
        question = PATTERN_QUESTIONS.get(pattern) or PATTERN_QUESTIONS.get(pattern.split(" / ")[0])
        if question:
            lines.append(f"\n**Concrete vraag om te stellen:**")
            lines.append(f"- \"{question}\"")

    return "\n".join(lines)


def build_motivational_close(metadata: Dict[str, Any]) -> str:
    """
    Rule 8: Short motivational close.
    """
    coverage = metadata.get("coverage_percentage", 0)
    prosody = metadata.get("prosody_score", 0)
    
    if coverage >= 70 and prosody >= THRESHOLDS["prosody"]["good"]:
        return "=== 7. Afsluiting ===\n\nJe hebt een solide basis gelegd. Blijf oefenen met de actiepunten en je zult nog sterker worden in je gespreksvaardigheden. Succes met de volgende oefening!"
    elif coverage >= 40:
        return "=== 7. Afsluiting ===\n\nJe maakt goede vooruitgang. Focus op de verbeterpunten en blijf vooral veel oefenen. Elke oefening maakt je beter!"
    else:
        return "=== 7. Afsluiting ===\n\nDit is een leerproces. Pak de actiepunten op en probeer het opnieuw. Met oefening wordt je steeds beter in het voeren van een goede anamnese."


def build_lecturer_notes_section(notes: Optional[str]) -> Optional[str]:
    if not notes:
        return None
    return "=== Optionele Docentnotities ===\n\n*" + scrub_text(notes) + "*"


def format_student_feedback(
    conversation_feedback,
    gordon_result,
    speech_result,
    conversation_history: Optional[str] = None,
):
    """
    Format all feedback components into a clear, student-friendly structure.
    Returns a dict so routes can pass only the plain text to TTS while still exposing
    structured metadata to the client.
    """
    # Build metadata first so we can measure coverage and delivery.
    metadata = build_feedback_metadata(gordon_result, speech_result)
    llm_prompt = metadata.get("llm_prompt") or build_llm_prompt(metadata)
    metadata["llm_prompt"] = llm_prompt
    conversation_analysis = build_conversation_analysis_block(conversation_history, gordon_result)
    metadata["conversation_analysis_block"] = conversation_analysis

    # Detect gaps tussen uitgesproken en getoond begrip.
    gap_result = analyze_understanding_gaps(conversation_history, metadata)
    if conversation_analysis.get("flags", {}).get("comprehension_gap"):
        gap_result["gap_detected"] = True
        existing_phrases = gap_result.get("exact_phrases", [])
        for phrase in conversation_analysis.get("extracted_examples", {}).get("understanding_examples", []):
            if phrase not in existing_phrases:
                existing_phrases.append(phrase)
        gap_result["exact_phrases"] = existing_phrases
        gap_result.setdefault("reasons", []).append("Transcript: begrip geclaimd zonder parafrase binnen 2 zinnen.")
    metadata["understanding_gap"] = gap_result

    summary_section = build_summary_section(metadata, conversation_history, conversation_analysis)

    llm_sections, lecturer_notes = sanitize_llm_output(conversation_feedback, metadata)
    conversation_skills_text = build_conversation_skills_section(llm_sections, metadata, conversation_analysis)

    speech_section_text = build_speech_section(speech_result, metadata, conversation_analysis)
    understanding_section_text = build_understanding_section(gap_result, conversation_analysis)
    gordon_section_text = build_gordon_section(metadata)
    action_items_text = build_action_items(metadata)
    lecturer_section_text = build_lecturer_notes_section(lecturer_notes)

    # Rule 8: Add motivational close
    motivational_close = build_motivational_close(metadata)
    
    ordered_sections: List[str] = [
        summary_section,
        conversation_skills_text,
        speech_section_text,
        understanding_section_text,
        gordon_section_text,
        action_items_text,
        motivational_close,
        lecturer_section_text,
    ]

    formatted_feedback = "\n\n".join(filter(None, ordered_sections))

    structured_sections: Dict[str, Any] = {
        "summary": summary_section,
        "gespreksvaardigheden": conversation_skills_text,
        "understanding_gap": understanding_section_text,
        "gordon": gordon_section_text,
        "action_items": action_items_text,
        "closing": motivational_close,
        "llm_sections": llm_sections,
        "analysis_block": conversation_analysis,
    }
    if speech_section_text:
        structured_sections["speech"] = speech_section_text
    if lecturer_section_text:
        structured_sections["lecturer_notes"] = lecturer_section_text

    structured_payload = {
        "sections": structured_sections,
        "metadata": metadata,
    }

    return {
        "text": formatted_feedback,
        "structured": structured_payload,
    }
