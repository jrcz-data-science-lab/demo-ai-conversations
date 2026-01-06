"""
Advanced formatter: produce structured, metric-driven feedback for nursing students.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from phase_detection import (
    DEFAULT_PHASE_CONFIG,
    analyze_conversation_phases,
    detect_gordon_patterns,
    looks_like_question,
    render_phase_feedback,
)

# Threshold configuration lives here to keep logic easy to tune.
THRESHOLDS = {
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
STUDENT_LINE = re.compile(r"^\s*student\b.*?:", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b")
ADDRESS_PATTERN = re.compile(
    r"(?:\b\d{1,4}[A-Za-z]?(?:-\d{1,2})?\s?(?:[A-Za-z]{2})?\s?(?:straat|laan|weg|dreef|road|rd|st|ave)\b)|"
    r"(?:\b[A-Za-z]{2,}\s(?:straat|laan|weg|dreef|road|rd|st|ave)\s?\d{1,4}[A-Za-z]?(?:-\d{1,2})?\b)",
    re.IGNORECASE,
)
POSTCODE_PATTERN = re.compile(r"\b\d{4}\s?[A-Z]{2}\b", re.IGNORECASE)

# Quick reference questions per Gordon-patroon to make follow-ups concrete.
PATTERN_QUESTIONS = {
    # English labels (from GORDON_PATTERNS).
    "Health Perception / Management": "Hoe ervaart u uw gezondheid op dit moment en welke zorg gebruikt u?",
    "Health Perception–Health Management": "Hoe ervaart u uw gezondheid op dit moment en welke zorg gebruikt u?",
    "Nutritional–Metabolic": "Hoe gaat het met eten en drinken; heeft u genoeg eetlust?",
    "Elimination": "Kunt u vertellen hoe het gaat met plassen en ontlasting?",
    "Activity–Exercise": "Hoe mobiel voelt u zich en wat lukt er in huis qua bewegen?",
    "Sleep–Rest": "Hoe slaapt u de laatste tijd en wordt u uitgerust wakker?",
    "Cognitive–Perceptual": "Merkt u veranderingen in uw geheugen, concentratie of waarneming?",
    "Self-Perception / Self-Concept": "Hoe voelt u zich over uzelf sinds de klachten zijn begonnen?",
    "Self-Perception–Self-Concept": "Hoe voelt u zich over uzelf sinds de klachten zijn begonnen?",
    "Role–Relationship": "Wie ondersteunt u thuis en hoe verloopt dat voor u?",
    "Roles–Relationships": "Wie ondersteunt u thuis en hoe verloopt dat voor u?",
    "Sexuality–Reproductive": "Heeft de situatie invloed op intimiteit of relaties?",
    "Coping–Stress Tolerance": "Wat doet u als het even tegenzit en wat helpt u te ontspannen?",
    "Values–Belief": "Zijn er overtuigingen of waarden die we moeten meenemen in uw zorg?",
    "Values–Beliefs": "Zijn er overtuigingen of waarden die we moeten meenemen in uw zorg?",
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
# These phrases kunnen het gesprek te snel afsluiten als ze niet gevolgd worden door parafrase of checkvraag.
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
COMPREHENSION_GAP_PHRASES = list(dict.fromkeys([p.strip() for p in COMPREHENSION_GAP_PHRASES if p.strip()]))

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

# Explicit understanding claims (used to trigger gaps when unsupported)
UNDERSTANDING_CLAIM_CUES = [
    "ik begrijp het",
    "ik snap het",
    "ik begrijp u",
    "ik begrijp je",
    "ik snap u",
    "ik snap je",
    "i understand",
    "i get it",
    "got it",
    "that makes sense",
    "it makes sense",
    "okay i understand",
    "ok i understand",
    "okay got it",
    "ok got it",
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


def scrub_text(text: Optional[Any]) -> str:
    """
    Remove potential identifying details from any string content.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    cleaned = NAME_PATTERN.sub("[patiënt]", text)
    cleaned = PHONE_PATTERN.sub("[privé-nummer]", cleaned)
    cleaned = ADDRESS_PATTERN.sub("[adres verwijderd]", cleaned)
    cleaned = POSTCODE_PATTERN.sub("[postcode verwijderd]", cleaned)
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
        if STUDENT_LINE.match(line):
            message = line.split(":", 1)[1].strip()
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


def conversation_history_to_turns(conversation_history: Optional[str]) -> List[Dict[str, Any]]:
    """
    Convert stored history string ("Speaker: text") into structured turns.
    """
    if not conversation_history:
        return []
    turns: List[Dict[str, Any]] = []
    for line in conversation_history.splitlines():
        if ":" not in line:
            continue
        speaker_raw, text = line.split(":", 1)
        speaker_norm = speaker_raw.strip().lower()
        speaker_label = "student" if speaker_norm.startswith("student") else "patient"
        turns.append({"speaker": speaker_label, "text": text.strip()})
    return turns


def gordon_stub_from_detection(gordon_detected: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convert deterministic Gordon detection output to the shape expected by metadata builder.
    """
    if not gordon_detected:
        return None
    patterns = gordon_detected.get("patterns", {}) or {}
    mentioned = []
    missing = []
    pattern_details: Dict[str, Dict[str, Any]] = {}
    for pid, pdata in patterns.items():
        name = pdata.get("name") or pid
        pattern_details[pid] = {"name": name}
        if pdata.get("covered"):
            mentioned.append(name)
        else:
            missing.append(name)

    return {
        "coverage_percentage": gordon_detected.get("coverage_percent", 0.0),
        "covered_patterns": gordon_detected.get("covered_count", 0),
        "total_patterns": gordon_detected.get("total_patterns", 11),
        "mentioned_patterns": mentioned,
        "missing_patterns": missing,
        "pattern_details": pattern_details,
        "summary": "",
    }


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
            "abrupt_ending_examples": [],
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
        is_open = any(cleaned.startswith(prefix + " ") or cleaned == prefix for prefix in ANALYSIS_OPEN_QUESTION_PREFIXES)
        is_closed = any(cleaned.startswith(prefix + " ") or cleaned == prefix for prefix in ANALYSIS_CLOSED_QUESTION_PREFIXES)
        if is_open:
            analysis_block["exact_phrases_used"]["open_questions"].append(sentence)
        elif is_closed:
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
    analysis_block["extracted_examples"]["abrupt_ending_examples"] = abrupt_examples

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


def _normalize_for_match(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_no_punct(s: str) -> str:
    s = _normalize_for_match(s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_exact_phrases(messages: List[str], phrases: List[str]) -> List[Tuple[str, int]]:
    """
    Find phrases used and their message index.
    Robust to punctuation differences by normalizing both message and phrase to no-punct.
    """
    found: List[Tuple[str, int]] = []
    seen = set()

    if not phrases:
        return []

    phrase_norm_map = []
    for p in phrases:
        p_no_punct = _normalize_no_punct(p)
        if not p_no_punct:
            continue
        phrase_norm_map.append((p, p_no_punct, p_no_punct.split()))

    for idx, msg in enumerate(messages):
        msg_norm = _normalize_for_match(msg)
        msg_no_punct = _normalize_no_punct(msg)

        for original_phrase, phrase_no_punct, words in phrase_norm_map:
            if not words:
                continue
            key = (phrase_no_punct, idx)
            if key in seen:
                continue

            # Single-word phrases: match on word boundary in either normalized form
            if len(words) == 1:
                w = re.escape(words[0])
                if re.search(rf"\b{w}\b", msg_no_punct, flags=re.IGNORECASE) or re.search(
                    rf"\b{w}\b", msg_norm, flags=re.IGNORECASE
                ):
                    found.append((original_phrase, idx))
                    seen.add(key)
                continue

            pattern = r"\b" + r"(?:\s|[^\w])+".join(map(re.escape, words)) + r"\b"
            if re.search(pattern, msg_no_punct, flags=re.IGNORECASE) or re.search(
                pattern, msg_norm, flags=re.IGNORECASE
            ):
                found.append((original_phrase, idx))
                seen.add(key)

    return found


def analyze_understanding_gaps(conversation_history: Optional[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect: student claims understanding WITHOUT demonstrating it via paraphrase or checkvraag within same/next 2 messages.
    """
    student_messages = extract_student_messages(conversation_history)
    if not student_messages:
        return {
            "gap_detected": False,
            "reasons": [],
            "exact_phrases": [],
            "gap_issues": [],
            "summary": "Geen studentuitspraken beschikbaar om begrip te toetsen.",
            "student_messages": 0,
            "understanding_statements": 0,
            "paraphrase_attempts": 0,
            "checkvraag_attempts": 0,
            "followup_questions": 0,
            "confusion_signals": 0,
        }

    lower_messages = [_normalize_for_match(m) for m in student_messages]

    def count_cues(cues: List[str]) -> int:
        total = 0
        for msg in student_messages:
            msg_norm = _normalize_no_punct(msg)
            for cue in cues:
                cue_norm = _normalize_no_punct(cue)
                if not cue_norm:
                    continue
                if len(cue_norm.split()) == 1 and len(cue_norm) <= 4:
                    if re.search(rf"\b{re.escape(cue_norm)}\b", msg_norm):
                        total += 1
                else:
                    if cue_norm in msg_norm:
                        total += 1
        return total

    gap_phrases_found = find_exact_phrases(student_messages, COMPREHENSION_GAP_PHRASES)

    gap_issues = []
    exact_phrases_quoted: List[str] = []

    for phrase, msg_idx in gap_phrases_found:
        phrase_norm = _normalize_for_match(phrase)
        current = lower_messages[msg_idx]

        def _has_evidence(text: str) -> bool:
            return any(cue in text for cue in PARAPHRASE_CUES) or any(cue in text for cue in CHECKVRAAG_CUES)

        followed_by_evidence = False

        pos = current.find(phrase_norm)
        if pos >= 0:
            after = current[pos + len(phrase_norm):]
            if _has_evidence(after):
                followed_by_evidence = True

        if not followed_by_evidence:
            next_window = " ".join(lower_messages[msg_idx + 1: msg_idx + 3])
            if _has_evidence(next_window):
                followed_by_evidence = True

        if not followed_by_evidence:
            exact_phrases_quoted.append(phrase)
            gap_issues.append(
                {
                    "phrase": phrase,
                    "message_index": msg_idx,
                    "exact_message": student_messages[msg_idx],
                }
            )

    understanding_hits = count_cues(UNDERSTANDING_CUES)
    claim_hits = count_cues(UNDERSTANDING_CLAIM_CUES)
    paraphrase_hits = count_cues(PARAPHRASE_CUES)
    checkvraag_hits = count_cues(CHECKVRAAG_CUES)
    confusion_hits = count_cues(CONFUSION_CUES)
    question_count = sum(1 for msg in student_messages if looks_like_question(msg))

    coverage = float(metadata.get("coverage_percentage", 0) or 0)
    missing_patterns = metadata.get("patterns_missing") or []

    gap_reasons: List[str] = []
    if gap_issues:
        unique = list(dict.fromkeys([g["phrase"] for g in gap_issues]))
        sample = "', '".join(unique[:3])
        if len(unique) > 3:
            sample += f"' (+{len(unique) - 3} meer)"
        gap_reasons.append(
            f"Begrip geclaimd met '{sample}' zonder parafrase/checkvraag binnen (max) 2 vervolgberichten."
        )

    if understanding_hits and (paraphrase_hits == 0 and checkvraag_hits == 0):
        gap_reasons.append("Er werd geen parafrase of checkvraag gebruikt om begrip te toetsen.")
    if understanding_hits and question_count < max(1, understanding_hits):
        gap_reasons.append("Na begrip-uitingen volgden relatief weinig verdiepende/verifiërende vragen.")
    if confusion_hits:
        gap_reasons.append("Er zijn signalen van twijfel/zoekend taalgebruik in je antwoorden.")
    if coverage < 60 and understanding_hits:
        focus = ", ".join(missing_patterns[:2]) if missing_patterns else "belangrijke patronen"
        gap_reasons.append(f"Let op: relevante patronen nog onbesproken ({focus}).")

    primary_gap = bool(gap_issues) or (claim_hits > 0 and (paraphrase_hits + checkvraag_hits) == 0)
    gap_detected = primary_gap
    summary = (
        " ".join(gap_reasons[:3])
        if gap_detected
        else "Je parafraseerde of stelde vervolgvragen waardoor begrip aannemelijk is."
    )

    return {
        "gap_detected": gap_detected,
        "reasons": gap_reasons,
        "exact_phrases": list(dict.fromkeys(exact_phrases_quoted)),
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
        return "zoekend"
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

    emotion_adjustment = {"calm": 4.0, "empathetic": 6.0, "neutral": 0.0, "zoekend": -4.0, "stressed": -6.0, "confused": -5.0}
    score = base - tempo_penalty - filler_penalty - pause_penalty + emotion_adjustment.get(emotion, 0.0)
    return max(0.0, min(100.0, round(score, 1)))


def build_feedback_metadata(gordon_result: Optional[Dict], speech_result: Optional[Dict]) -> Dict[str, Any]:
    """
    Compile reusable metadata that can power prompt generation and UI rendering.
    """
    gordon_result = gordon_result or {}
    speech_result = speech_result or {}
    metrics = speech_result.get("metrics", {}) or {}
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
        "speech_summary": scrub_text((speech_result or {}).get("summary")),
        "emotion": infer_emotion((speech_result or {}).get("emotion"), metrics),
        "coverage_percentage": round(gordon_result.get("coverage_percentage", 0)),
        "student_coverage_percentage": round(gordon_result.get("student_coverage_percent", gordon_result.get("coverage_percentage", 0))),
        "patterns_mentioned": mentioned_pattern_labels,
        "patterns_missing": missing_pattern_labels,
        "covered_patterns": gordon_result.get("covered_patterns", 0),
        "student_covered_patterns": gordon_result.get("student_covered_count", gordon_result.get("covered_patterns", 0)),
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
    allowed_quotes_text = format_allowed_quotes(metadata.get("allowed_quotes") or {})
    coverage_any = metadata.get("coverage_percentage", 0)
    coverage_student = metadata.get("student_coverage_percentage", coverage_any)

    prompt = f"""
Je bent een professionele beoordelaar van gespreksvaardigheden voor HBO-V studenten.
Gebruik ALTIJD de onderstaande structuur en schrijf in duidelijk, vriendelijk en professioneel Nederlands.
Verwijs naar de METRICS (onderaan) om je feedback concreet en specifiek te maken.
Vermijd vage opmerkingen, wees feitelijk, empathisch en gericht op leerdoelen.
Als er geen citaten beschikbaar zijn in het transcript, schrijf expliciet \"Geen exacte quote beschikbaar\" en verzin niets.

BELANGRIJKE REGELS:
- Als dekking < 27% (minder dan 3/11 patronen): DE toon MOET kritisch zijn. Geef maximaal 1 compliment en focus op wat ontbreekt.
- Quote ALTIJD exacte studentuitdrukkingen (bijv. "Je zei: 'ik snap het'...").
- Noem exacte fillers als die voorkomen (bijv. "Je gebruikte 'eh' meerdere keren").
- Wees realistisch: bij lage dekking of lage prosodie moet de toon kritischer zijn.

CITATEN (ALLEEN DEZE MAG JE QUOTEN):
{allowed_quotes_text}

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
Gordon dekking (door student uitgevraagd): {coverage_student}%
Gordon dekking (alles wat ter sprake kwam): {coverage_any}%
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
            f"- Tonaliteit: {metadata['emotion']} | Prosodie {metadata['prosody_score']}/100."
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
        for line in sanitized_text.splitlines():
            header_line = line.strip().lower()
            header_line = re.sub(r"^[#\s>*_-]+", "", header_line)
            header_line = re.sub(r"^\d+[\).\s-]+", "", header_line)
            header_line = header_line.strip(": ").strip()
            matched = next(
                (
                    title
                    for title in LLM_SECTION_TITLES
                    if re.search(rf"\b{re.escape(title.lower())}\b", header_line)
                ),
                None,
            )
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
    coverage_any = metadata["coverage_percentage"]
    coverage_student = metadata.get("student_coverage_percentage", coverage_any)
    covered_patterns_any = metadata.get("covered_patterns", 0)
    covered_patterns_student = metadata.get("student_covered_patterns", covered_patterns_any)
    prosody = metadata["prosody_score"]
    gap_result = metadata.get("understanding_gap") or {}
    has_gap = gap_result.get("gap_detected")

    # Rule 4: Low coverage handling (< 3/11 patterns = ~27%), based on student-driven dekking
    is_low_coverage = covered_patterns_student < 3 or coverage_student < 27.3
    
    # Rule 5: Check if history is very short
    sentence_count = count_sentences(conversation_history) if conversation_history else 0
    is_very_short = sentence_count < 5

    # Determine overall level - Rule 4: Force orange/red for low student coverage
    if is_low_coverage:
        overall_level = "low"
        missing_patterns = metadata.get("patterns_missing", [])
        missing_list = ", ".join(missing_patterns[:3]) if missing_patterns else "belangrijke patronen"
        overall_reason = (
            f"Door slechts {covered_patterns_student} patronen zelf uit te vragen, mis je belangrijke informatie "
            f"die essentieel is voor een veilige anamnese. Ontbrekende patronen: {missing_list}."
        )
    elif coverage_student >= 70 and prosody >= THRESHOLDS["prosody"]["good"] and not has_gap:
        overall_level = "high"
        overall_reason = "Goede student-gedreven dekking, sterke prosodie en geen duidelijke begripskloof."
    elif coverage_student >= 40 and prosody >= THRESHOLDS["prosody"]["ok"]:
        overall_level = "medium"
        overall_reason = "Redelijke student-gedreven dekking of prosodie, maar er is ruimte voor verdieping of scherpere opvolging."
    else:
        overall_level = "low" if coverage_student < 40 or prosody < THRESHOLDS["prosody"]["ok"] else "medium"
        reason_parts = []
        if coverage_student < 40:
            reason_parts.append(f"lage student-dekking ({coverage_student}%)")
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

    # Report both any-mention and student-driven coverage
    if coverage_student >= 70:
        lines.append(
            f"- ✅ Gordon patronen (door student uitgevraagd): {covered_patterns_student}/{metadata['total_patterns']} "
            f"({coverage_student}%) behandeld."
        )
    elif coverage_student >= 40:
        lines.append(
            f"- ⚠️ Gordon patronen (door student uitgevraagd): {covered_patterns_student}/{metadata['total_patterns']} "
            f"({coverage_student}%) – pak meer domeinen mee."
        )
    else:
        lines.append(
            f"- ❌ Gordon patronen (door student uitgevraagd): slechts {coverage_student}% dekking, plan bewuste vervolgvragen."
        )
        if analysis_flags.get("low_coverage_issue"):
            lines.append(
                f"- ❌ Je vroeg slechts actief naar {covered_patterns_student} van de 11 patronen; vul de ontbrekende patronen aan met gerichte vragen."
            )

    if coverage_any != coverage_student:
        lines.append(
            f"- ℹ️ In totaal kwamen {covered_patterns_any}/{metadata['total_patterns']} patronen ter sprake "
            f"({coverage_any}%), maar niet alle patronen werden door jou zelf uitgevraagd."
        )

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
    if analysis_flags.get("abrupt_ending_issue") and analysis_examples.get("abrupt_ending_examples"):
        lines.append(f"- ❌ Afsluiting: '{analysis_examples['abrupt_ending_examples'][0]}' klonk abrupt en kwam zonder samenvatting.")

    lines.append(
        f"- Metrics: {metadata['speech_rate_wpm']} wpm | tempo-variatie {metadata['tempo_variation']}% | "
        f"pauze {metadata['pause_avg']}s | prosodie {metadata['prosody_score']}/100 | emotie {metadata['emotion']}."
    )
    return "\n".join(lines)


def format_pause_distribution_text(pause_distribution: Dict[str, float]) -> str:
    return f"{pause_distribution['short']}% kort / {pause_distribution['medium']}% middel / {pause_distribution['long']}% lang"


def format_allowed_quotes(allowed_quotes: Dict[str, List[str]]) -> str:
    """
    Render a constrained list of quotable transcript snippets to avoid hallucinated quotes.
    """
    def render_line(label: str, items: List[str]) -> str:
        if not items:
            return f"- {label}: Geen exacte uitspraak beschikbaar in het transcript."
        safe_items = [f"\"{scrub_text(q)}\"" for q in items[:3]]
        return f"- {label}: " + " | ".join(safe_items)

    sections = [
        render_line("Begrip-uitspraken", allowed_quotes.get("understanding") or []),
        render_line("Filler-voorbeelden", allowed_quotes.get("fillers") or []),
        render_line("Voorbeelden open vragen", allowed_quotes.get("open_questions") or []),
        render_line("Voorbeelden gesloten vragen", allowed_quotes.get("closed_questions") or []),
    ]
    return "\n".join(sections)


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
        lines.append("  → Je sprak erg langzaam; houd het tempo levendig zodat de patiënt alert blijft.")
    
    lines.append(f"- Tempo-variatie: {metadata['tempo_variation']}%")
    lines.append(f"- Pauzedistributie: {pause_text}")
    lines.append(f"- Gemiddelde pauzeduur: {metadata['pause_avg']} s")
    
    # Rule 2: Filler/hesitation detection - MANDATORY if any fillers found
    metrics = (speech_result or {}).get("metrics", {}) if speech_result else {}
    filler_tokens = (analysis_block.get("exact_phrases_used") or {}).get("filler_words", [])
    filler_token_count = len(filler_tokens)
    hesitation_markers = metadata.get("hesitation_markers", 0)
    filler_ratio = metadata.get('filler_ratio', 0)
    total_words = metadata.get('total_words', 0) or metrics.get('total_words', 0) or analysis_block.get("summary_findings", {}).get("conversation_length", 0)

    if filler_token_count > 0:
        filler_density = (filler_token_count / total_words * 10) if total_words > 0 else 0
        filler_summary = summarize_filler_tokens(filler_tokens)
        lines.append(f"- Opvulgeluidjes/fillers: {filler_summary}")
        lines.append(f"- Filler-dichtheid: {filler_density:.1f} per 10 woorden")
        if filler_token_count > 3:
            lines.append(f"  → Je gebruikte veel opvulgeluidjes ({filler_token_count} keer), wat de professionaliteit verlaagt.")
    else:
        lines.append("- Opvulgeluidjes/fillers: geen gedetecteerd in transcript")

    if hesitation_markers:
        lines.append(f"- Hesitation markers (model): {hesitation_markers}")

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

    return "\n".join(lines)


def build_conversation_skills_section(
    metadata: Dict[str, Any],
    conversation_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Section 2: Gespreksvaardigheden – grounded in transcript analysis.
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
    if flags.get("abrupt_ending_issue") and examples.get("abrupt_ending_examples"):
        lines.append(f"- ❌ Abrupte afsluiting: \"{examples['abrupt_ending_examples'][0]}\" zonder samenvatting.")

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
    lines = ["### Begripstoetsing (Comprehension Checking)"]
    analysis_block = conversation_analysis or {}
    analysis_flags = analysis_block.get("flags", {})
    analysis_examples = analysis_block.get("extracted_examples", {})
    analysis_phrases = analysis_block.get("exact_phrases_used", {})
    understanding_examples = analysis_examples.get("understanding_examples") or []
    gap_result = gap_result or {}

    gap_issues = gap_result.get("gap_issues") or []
    gap_phrases: List[str] = [issue.get("phrase") for issue in gap_issues if issue.get("phrase")]

    for source in (
        gap_result.get("exact_phrases") or [],
        understanding_examples,
    ):
        for phrase in source:
            if phrase and phrase not in gap_phrases:
                gap_phrases.append(phrase)

    if gap_phrases:
        lines.append("Gehoorde uitspraken die begrip claimen zonder toets:")
        for phrase in gap_phrases:
            lines.append(f"• \"{phrase}\"")
        lines.append("Deze uitdrukkingen klinken empathisch, maar tonen geen echte begripstoetsing. De patiënt kan hierdoor denken dat verdere uitleg niet nodig is.")
        lines.append("Verbeter met:")
        lines.append("• Gebruik parafrases zoals \"Dus u geeft aan dat...?\" om te laten horen wat jij hebt opgepikt.")
        lines.append("• Stel verduidelijkende of checkvragen, bijv. \"Wanneer merkt u dat vooral?\"")
        lines.append("• Vermijd automatische instemmers zoals \"ja ja\", \"is goed\" of \"ik snap het\"; geef liever een korte samenvatting.")
    else:
        lines.append("Je hebt goed begrip getoetst door te parafraseren en verduidelijkende vragen te stellen.")

    return "\n".join(lines)


def build_comprehension_section(comprehension_phrases):
    """
    Build the 'Begripstoetsing' section based on the detected comprehension gap phrases.
    """
    if not comprehension_phrases:
        return (
            "### Begripstoetsing (Comprehension Checking)\n"
            "Goed gedaan: je hebt begrip actief getoetst door te parafraseren of verduidelijkende vragen te stellen.\n\n"
        )

    section = "### Begripstoetsing (Comprehension Checking)\n"
    section += "We detecteerden de volgende uitdrukkingen die géén echte begripstoetsing laten zien:\n"

    for phrase in comprehension_phrases:
        section += f"- \"{phrase}\"\n"

    section += (
        "\nDeze zinnen klinken empathisch, maar ze stoppen het gesprek omdat de patiënt kan denken "
        "dat verdere uitleg niet nodig is. Dit belemmert klinisch redeneren.\n\n"
        "**Wat helpt meer?**\n"
        "- Parafraseren: \"Dus u bedoelt dat…?\"\n"
        "- Doorvragen: \"Wanneer merkt u dat vooral?\"\n"
        "- Vermijd automatische reacties zoals \"ik begrijp het\", \"is goed\", \"precies\", \"ja ja\".\n\n"
    )

    return section


def build_gordon_section(metadata: Dict[str, Any]) -> str:
    """
    Describe Gordon pattern coverage and missing aspects.
    Rule 4: If coverage < 3/11, explicitly list missing patterns and explain why they're essential.
    """
    lines = ["=== 5. Gordon-patronen Analyse ==="]
    detected = metadata.get("detected_gordon")
    if detected:
        coverage_any = detected.get("coverage_percent", 0.0)
        covered_patterns_any = detected.get("covered_count", 0)
        coverage_student = detected.get("student_coverage_percent", coverage_any)
        covered_patterns_student = detected.get("student_covered_count", covered_patterns_any)
        total_patterns = detected.get("total_patterns", 11)
        patterns = detected.get("patterns", {}) or {}
        mentioned_list = [p.get("name") for p in patterns.values() if p.get("covered")]
        missing_list = [p.get("name") for p in patterns.values() if not p.get("covered")]
    else:
        coverage_any = metadata["coverage_percentage"]
        covered_patterns_any = metadata['covered_patterns']
        coverage_student = metadata.get("student_coverage_percentage", coverage_any)
        covered_patterns_student = metadata.get("student_covered_patterns", covered_patterns_any)
        total_patterns = metadata['total_patterns']
        mentioned_list = metadata.get("patterns_mentioned") or []
        missing_list = metadata.get("patterns_missing") or []

    lines.append(
        f"- Dekking (door student uitgevraagd): {covered_patterns_student}/{total_patterns} ({coverage_student:.1f}%)"
    )
    lines.append(
        f"- Totale dekking (alles wat ter sprake kwam): {covered_patterns_any}/{total_patterns} ({coverage_any:.1f}%)"
    )

    # Rule 4: Strong feedback for low coverage
    is_low_coverage = covered_patterns_student < 3 or coverage_student < 27.3
    if is_low_coverage:
        lines.append(
            f"- ⚠️ Lage dekking: Door slechts {covered_patterns_student} patronen actief uit te vragen, mis je belangrijke informatie "
            "die essentieel is voor een veilige anamnese."
        )
        lines.append(
            f"- ❌ Je behandelde als student slechts {covered_patterns_student} van de 11 patronen; zorg voor een bredere anamnese."
        )

    mentioned = ", ".join(mentioned_list) if mentioned_list else "geen vermeld"
    missing = ", ".join(missing_list) if missing_list else "n.v.t."
    lines.append(f"- Genoemde patronen: {mentioned}")
    
    # Rule 4: Explicitly list missing patterns for low coverage
    if is_low_coverage and missing_list:
        lines.append(f"- ❌ Ontbrekende patronen (essentieel): {missing}")
    else:
        lines.append(f"- Ontbrekende patronen: {missing}")

    if metadata["gordon_summary"]:
        lines.append(f"- Samenvatting: {metadata['gordon_summary']}")

    top_missing = ", ".join(missing_list[:3]) if missing_list else "n.v.t."
    if top_missing != "n.v.t.":
        lines.append(f"- Focus voor volgende keer: {top_missing}")

    # Add concrete follow-up questions for the most relevant missing patterns.
    follow_ups = []
    for pattern in missing_list[:2]:
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
    filler_token_count = len(analysis_exact.get("filler_words", []))
    hesitation_markers = metadata.get("hesitation_markers", 0)
    tempo_variation = metadata["tempo_variation"]
    speech_rate = metadata["speech_rate_wpm"]
    pause_distribution = metadata["pause_distribution"]
    pause_avg = metadata["pause_avg"]
    emotion = metadata["emotion"]
    hesitations = metadata["hesitation_markers"]
    prosody = metadata["prosody_score"]
    volume_stability = metadata["volume_stability"]
    gap_result = metadata.get("understanding_gap") or {}
    coverage = metadata.get("student_coverage_percentage", metadata.get("coverage_percentage", 0))
    covered_count = metadata.get("student_covered_patterns", metadata.get("covered_patterns", 0))
    
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
        improvements.append("Veel korte pauzes breken de flow; adem diep uit voordat je reageert.")
    # Rule 2B: Filler reduction action
    if filler_token_count > 3:
        improvements.append("Verminder het aantal fillers zoals 'eh', omdat dit de boodschap minder helder maakt. Gebruik korte pauzes in plaats van opvulgeluidjes.")
    elif filler_ratio > THRESHOLDS["filler_ratio"]["medium"]:
        improvements.append("Verminder opvulgeluidjes door steekwoorden vooraf te noteren en rustiger te spreken.")
    elif analysis_flags.get("filler_issue") and analysis_exact.get("filler_words"):
        filler_summary = summarize_filler_tokens(analysis_exact.get("filler_words", []))
        improvements.append(f"Schrap de fillers {filler_summary} uit je vragen; neem een korte stilte in plaats van 'eh'.")
    if emotion in {"zoekend", "stressed"}:
        improvements.append("Je klonk wat zoekend; vertraag je ademhaling en vat antwoorden samen om rust te brengen.")
    if prosody < THRESHOLDS["prosody"]["ok"]:
        improvements.append("Werk aan vocale variatie door sleutelwoorden te benadrukken en toonhoogte licht te variëren.")
    if volume_stability and volume_stability < THRESHOLDS["volume_stability"]:
        improvements.append("Houd je volume stabiel door rechtop te zitten en uit te ademen tijdens het spreken.")
    if coverage < 60:
        missing = ", ".join(metadata["patterns_missing"][:3])
        improvements.append(f"Plan vragen rond ontbrekende patronen ({missing}) om vollediger te screenen.")
    if analysis_flags.get("abrupt_ending_issue") and analysis_examples.get("abrupt_ending_examples"):
        improvements.append(f"Sluit niet af met \"{analysis_examples['abrupt_ending_examples'][0]}\"; eindig met een korte samenvatting en een bedankje.")
    if gap_result.get("gap_detected"):
        exact_phrases = gap_result.get("exact_phrases", [])
        if exact_phrases:
            phrases_example = "', '".join(exact_phrases[:3])
            improvements.append(
                f"Vermijd uitspraken zoals '{phrases_example}' als afsluiter zonder vervolg. "
                "Zonder parafrase of checkvraag kan dit het gesprek voortijdig sluiten. "
                "Doe liever: parafrase ('Dus u zegt dat...?') + checkvraag ('Heb ik dat goed begrepen?')."
            )
        else:
            improvements.append(
                "Vermijd uitspraken zoals 'ik begrijp het', 'ja ik snap het', 'i understand', 'i get it' als afsluiter. "
                "Zonder parafrase of checkvraag kan dit het gesprek vroegtijdig stoppen. "
                "Gebruik liever een parafrase ('Dus u zegt dat...?') en checkvraag ('Heb ik dat goed begrepen?')."
            )
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
            techniques.append(
                f"Techniek: Vermijd uitspraken zoals '{phrases_example}' zonder dat je daarna samenvat of een checkvraag stelt. "
                "Laat liever horen wat je hebt opgepikt ('Dus u voelt zich...?') en vraag of dat klopt."
            )
        else:
            techniques.append(
                "Techniek: Vermijd uitspraken zoals 'ik begrijp het', 'i understand', 'i get it' als afsluiter. "
                "Toon begrip door kort samen te vatten ('Dus u zegt dat...') en af te checken of dat klopt."
            )
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


def build_allowed_quotes(conversation_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Collect a limited set of transcript snippets the LLM is allowed to quote.
    """
    extracted = (conversation_analysis or {}).get("extracted_examples") or {}
    exact = (conversation_analysis or {}).get("exact_phrases_used") or {}
    return {
        "understanding": extracted.get("understanding_examples") or [],
        "fillers": extracted.get("filler_examples") or [],
        "open_questions": exact.get("open_questions") or [],
        "closed_questions": exact.get("closed_questions") or [],
    }


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
    turns = conversation_history_to_turns(conversation_history)
    phase_analysis = analyze_conversation_phases(turns, DEFAULT_PHASE_CONFIG)
    detected_gordon = detect_gordon_patterns(turns, DEFAULT_PHASE_CONFIG)
    gordon_stub = gordon_stub_from_detection(detected_gordon) or gordon_result

    # Build metadata first so we can measure coverage and delivery.
    metadata = build_feedback_metadata(gordon_stub, speech_result)
    metadata["detected_gordon"] = detected_gordon
    metadata["phase_analysis"] = phase_analysis
    metadata["gordon_model_result"] = gordon_result
    conversation_analysis = build_conversation_analysis_block(conversation_history, gordon_stub)
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

    if detected_gordon:
        metadata["coverage_percentage"] = detected_gordon.get("coverage_percent", metadata.get("coverage_percentage", 0))
        metadata["student_coverage_percentage"] = detected_gordon.get(
            "student_coverage_percent", metadata.get("student_coverage_percentage", metadata.get("coverage_percentage", 0))
        )
        metadata["covered_patterns"] = detected_gordon.get("covered_count", metadata.get("covered_patterns", 0))
        metadata["student_covered_patterns"] = detected_gordon.get(
            "student_covered_count", metadata.get("student_covered_patterns", metadata.get("covered_patterns", 0))
        )
        metadata["total_patterns"] = detected_gordon.get("total_patterns", metadata.get("total_patterns", 11))
        detected_patterns = detected_gordon.get("patterns", {}) or {}
        metadata["patterns_missing"] = [
            info.get("name")
            for info in detected_patterns.values()
            if not info.get("covered")
        ]
        metadata["patterns_mentioned"] = [
            info.get("name")
            for info in detected_patterns.values()
            if info.get("covered")
        ]
    combined_metrics = (phase_analysis.get("metrics") or {}).copy()
    if detected_gordon:
        combined_metrics["gordon_covered_count"] = detected_gordon.get("covered_count", 0)
        combined_metrics["gordon_coverage_percent"] = detected_gordon.get("coverage_percent", 0.0)
    phase_output = {
        "phases": phase_analysis.get("phases", {}),
        "metrics": combined_metrics,
        "gordon": detected_gordon,
        "evidence": phase_analysis.get("evidence", []),
    }
    metadata["phase_output"] = phase_output

    # Build allowed quotes list for the LLM prompt and regenerate the prompt with these constraints.
    metadata["allowed_quotes"] = build_allowed_quotes(conversation_analysis)
    metadata["llm_prompt"] = build_llm_prompt(metadata)

    summary_section = build_summary_section(metadata, conversation_history, conversation_analysis)

    llm_sections, lecturer_notes = sanitize_llm_output(conversation_feedback, metadata)
    conversation_skills_text = build_conversation_skills_section(metadata, conversation_analysis)

    comprehension_phrases: List[str] = []
    for source in (
        gap_result.get("exact_phrases") or [],
        (conversation_analysis.get("extracted_examples") or {}).get("understanding_examples", []),
        (conversation_analysis.get("exact_phrases_used") or {}).get("understanding_phrases", []),
    ):
        for phrase in source:
            if phrase and phrase not in comprehension_phrases:
                comprehension_phrases.append(phrase)
    comprehension_section_text = build_comprehension_section(comprehension_phrases)
    phase_feedback_text = render_phase_feedback(phase_analysis)

    speech_section_text = build_speech_section(speech_result, metadata, conversation_analysis)
    gordon_section_text = build_gordon_section(metadata)
    action_items_text = build_action_items(metadata)
    lecturer_section_text = build_lecturer_notes_section(lecturer_notes)

    # Rule 8: Add motivational close
    motivational_close = build_motivational_close(metadata)
    
    ordered_sections: List[str] = [
        summary_section,
        conversation_skills_text,
        comprehension_section_text,
        phase_feedback_text,
        speech_section_text,
        gordon_section_text,
        action_items_text,
        motivational_close,
        lecturer_section_text,
    ]

    formatted_feedback = "\n\n".join(filter(None, ordered_sections))

    structured_sections: Dict[str, Any] = {
        "summary": summary_section,
        "gespreksvaardigheden": conversation_skills_text,
        "comprehension": comprehension_section_text,
        "phase_feedback": phase_feedback_text,
        "gordon": gordon_section_text,
        "action_items": action_items_text,
        "closing": motivational_close,
        "llm_sections": llm_sections,
        "analysis_block": conversation_analysis,
    }
    structured_sections["phase_analysis"] = phase_analysis
    structured_sections["phase_output"] = phase_output
    if detected_gordon:
        structured_sections["gordon_detected"] = detected_gordon
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

def print_feedback_to_terminal(feedback_json):
    """
    Pretty-print key parts of the feedback JSON for terminal/server logs.
    Works for both client.py and app.py.
    """
    print("\n================ FEEDBACK SUMMARY ================\n")

    # 1. Main text
    text = feedback_json.get("response") or feedback_json.get("text")
    if text:
        print("📝 FEEDBACK TEXT:\n")
        print(text)
        print("\n---------------------------------------------------\n")

    # 2. Speech metrics
    metrics = feedback_json.get("speech_metrics")
    if metrics:
        print("📊 SPEECH METRICS:")
        print(f"- Speech Rate: {metrics.get('speech_rate_wpm', 'N/A')} wpm")
        print(f"- Avg Pause: {metrics.get('avg_pause', 'N/A')}s")
        print(f"- Filler Count: {metrics.get('filler_count', 'N/A')}")
        print(f"- Filler Ratio: {metrics.get('filler_ratio', 'N/A')}%")
        print()

    # 3. Icon states (for your UE model)
    icons = feedback_json.get("icon_states")
    if icons:
        print("🎛 ICON STATES:")
        print(f"- Speech Rate: {icons.get('speech_rate', 'N/A')}")
        print(f"- Pauses: {icons.get('pauses', 'N/A')}")
        print(f"- Fillers: {icons.get('fillers', 'N/A')}")
        print(f"- Overall: {icons.get('overall', 'N/A')}")
        print()

    # 4. Gordon pattern analysis
    gp = feedback_json.get("gordon_patterns")
    if gp:
        print("📚 GORDON PATTERN ANALYSIS:")
        print(f"- Coverage: {gp.get('covered_patterns', 0)}/{gp.get('total_patterns', 11)} "
              f"({gp.get('coverage_percentage', 0)}%)")
        print(f"- Patterns Mentioned: {gp.get('mentioned_patterns', [])}")
        print(f"- Summary: {gp.get('summary', '')}")
        print()

    # 5. Structured feedback entry point
    if "structured" in feedback_json:
        print("📑 STRUCTURED FEEDBACK AVAILABLE")
        print("(This includes all detailed sections generated by your formatter.)")
        print()

    print("===================================================\n")
