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
    # Dutch labels
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

# The comprehension-gap phrase list remains unchanged
COMPREHENSION_GAP_PHRASES = [
    # Dutch phrases...
    "ik begrijp het", "ik snap het", "ik begrijp u", "ik begrijp je", "ik snap u",
    "ik snap je", "ja, ik snap het", "ja ik snap het", "ja, ik begrijp het",
    "ja ik begrijp het", "ik weet het", "ja, ik weet het", "ja ik weet het",
    "oké, duidelijk", "oke, duidelijk", "duidelijk",
    "oké, ik begrijp het", "oke, ik snap het",
    "we gaan het regelen", "is goed", "ja dat klopt", "dat klopt",
    "ja precies", "precies", "klopt", "akkoord", "natuurlijk", "inderdaad",
    "ah oké", "ah oke", "ah duidelijk", "oké oké", "oke oke",

    # English phrases...
    "i understand", "i get it", "i understand you", "yes i understand",
    "yes, i understand", "yeah i understand", "got it", "i see",
    "yes i see", "yes, i see", "yeah i see", "that's right",
    "correct", "yes exactly", "exactly", "that makes sense",
]

UNDERSTANDING_CUES = [
    "ik begrijp", "ik snap", "als ik het goed begrijp", "dus u", "dus je",
    "i understand", "i get it", "i see", "understood", "got it"
]

PARAPHRASE_CUES = [
    "je geeft aan", "u geeft aan", "wat ik hoor", "dus u zegt", "dus je zegt",
    "met andere woorden", "samenvattend", "in other words",
]

CHECKVRAAG_CUES = [
    "heb ik dat goed", "klopt dat", "begrijp ik u goed",
    "did i understand correctly", "is that correct", "did i get that right",
]

CONFUSION_CUES = ["ik weet niet", "niet zeker", "twijfel", "uh", "uhm", "hmm"]

ANALYSIS_UNDERSTANDING_PHRASES = [
    "ik begrijp het", "ik snap het", "ik begrijp u", "oke duidelijk",
]
ANALYSIS_FILLER_WORDS = ["eh", "ehm", "uh", "uhm", "hmm"]
ANALYSIS_ABRUPT_CLOSINGS = ["ok dank u", "oke dank u", "oke doei", "ok bye"]
ANALYSIS_PARAPHRASE_CUES = ["dus u", "als ik het goed begrijp", "bedoelt u dat"]
ANALYSIS_OPEN_QUESTION_PREFIXES = ["wat", "hoe", "waar", "kunt u vertellen"]
ANALYSIS_CLOSED_QUESTION_PREFIXES = ["heb", "is", "kan", "kunt u", "bent u"]
ANALYSIS_SUMMARY_CUES = ["samenvattend", "kortom", "samengevat"]
EMPATHY_CUES = ["spijtig", "sorry", "vervelend", "kan me voorstellen"]

def scrub_text(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = NAME_PATTERN.sub("[patiënt]", text)
    cleaned = PHONE_PATTERN.sub("[privé-nummer]", cleaned)
    cleaned = ADDRESS_PATTERN.sub("[adres verwijderd]", cleaned)
    cleaned = re.sub(r"\b\d{9}\b", "[id verwijderd]", cleaned)
    return cleaned.strip()

def extract_student_messages(conversation_history: Optional[str]) -> List[str]:
    if not conversation_history:
        return []
    messages = []
    for line in conversation_history.splitlines():
        if line.strip().startswith("Student:"):
            msg = line.split("Student:", 1)[1].strip()
            if msg:
                messages.append(msg)
    return messages

def split_student_sentences(student_messages: List[str]) -> List[str]:
    sentences = []
    for msg in student_messages:
        parts = re.split(r'(?<=[.!?])\s+|\n+', msg)
        for p in parts:
            t = p.strip()
            if t:
                sentences.append(t)
    return sentences

def summarize_filler_tokens(tokens: List[str]) -> str:
    if not tokens:
        return ""
    c = {}
    for t in tokens:
        c[t] = c.get(t, 0) + 1
    return ", ".join(f"{t} ({c[t]}x)" for t in sorted(c.keys()))

def build_conversation_analysis_block(
    conversation_history: Optional[str],
    gordon_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    analysis_block = {
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
        return analysis_block

    student_messages = extract_student_messages(conversation_history)
    if not student_messages:
        return analysis_block

    sentences = split_student_sentences(student_messages)
    if not sentences:
        return analysis_block

    analysis_block["summary_findings"]["conversation_length"] = sum(len(s.split()) for s in sentences)
    lower_sentences = [s.lower() for s in sentences]

    paraphrase_indices = []
    empathy_attempts = []
    summary_indices = set()

    # FIRST PASS: Paraphrase cues, empathy, summaries
    for idx, sentence in enumerate(sentences):
        low = lower_sentences[idx]
        if any(cue in low for cue in ANALYSIS_PARAPHRASE_CUES):
            analysis_block["exact_phrases_used"]["paraphrases"].append(sentence)
            paraphrase_indices.append(idx)
        if any(cue in low for cue in EMPATHY_CUES):
            empathy_attempts.append(sentence)
        if any(cue in low for cue in ANALYSIS_SUMMARY_CUES):
            summary_indices.add(idx)

    analysis_block["exact_phrases_used"]["empathy_attempts"] = empathy_attempts

    # SECOND PASS: open vs closed questions
    for sentence in sentences:
        cleaned = re.sub(r"^[^a-zA-Z0-9]+", "", sentence.strip().lower())
        if any(cleaned.startswith(p + " ") or cleaned == p for p in ANALYSIS_OPEN_QUESTION_PREFIXES):
            analysis_block["exact_phrases_used"]["open_questions"].append(sentence)
        if any(cleaned.startswith(p + " ") or cleaned == p for p in ANALYSIS_CLOSED_QUESTION_PREFIXES):
            analysis_block["exact_phrases_used"]["closed_questions"].append(sentence)

    # THIRD PASS: comprehension gap (understanding phrases)
    for idx, sentence in enumerate(sentences):
        low = lower_sentences[idx]
        matched = [phrase for phrase in ANALYSIS_UNDERSTANDING_PHRASES if phrase in low]
        for phrase in matched:
            analysis_block["exact_phrases_used"]["understanding_phrases"].append(phrase)
            window = range(idx, min(len(sentences), idx + 3))
            if not any(pi in window for pi in paraphrase_indices):
                analysis_block["flags"]["comprehension_gap"] = True
                if sentence not in analysis_block["extracted_examples"]["understanding_examples"]:
                    analysis_block["extracted_examples"]["understanding_examples"].append(sentence)

    # FOURTH PASS: fillers
    filler_tokens = []
    filler_examples = []
    for sentence in sentences:
        tokens = re.findall(r"[\w']+", sentence.lower())
        found = False
        for token in tokens:
            if token in ANALYSIS_FILLER_WORDS:
                filler_tokens.append(token)
                found = True
        if found:
            filler_examples.append(sentence)

    analysis_block["exact_phrases_used"]["filler_words"] = filler_tokens
    analysis_block["extracted_examples"]["filler_examples"] = filler_examples

    if len(filler_tokens) > 1:
        analysis_block["flags"]["filler_issue"] = True

    # FIFTH PASS: abrupt endings
    abrupt_examples = []
    for idx, sentence in enumerate(sentences):
        low = lower_sentences[idx]
        if any(phrase in low for phrase in ANALYSIS_ABRUPT_CLOSINGS):
            analysis_block["exact_phrases_used"]["abrupt_closings"].append(sentence)
            nearby = any(j in summary_indices or j in paraphrase_indices
                         for j in range(max(0, idx-2), min(len(sentences), idx+1)))
            if not nearby:
                analysis_block["flags"]["abrupt_ending_issue"] = True
                abrupt_examples.append(sentence)

    analysis_block["extracted_examples"]["abrubt_ending_examples"] = abrupt_examples

    # Build issue summary list
    issues = []
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
    Detect comprehension gap: student claims to understand but does not paraphrase or ask clarification.
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
            "checkvraag_attempts": 0,
            "followup_questions": 0,
            "confusion_signals": 0,
        }

    lower_msgs = [m.lower() for m in student_messages]

    def count_hits(cues):
        return sum(1 for msg in lower_msgs for cue in cues if cue in msg)

    def find_exact(messages, phrases):
        found = []
        seen = set()
        for idx, msg in enumerate(messages):
            low = msg.lower()
            for phrase in phrases:
                ph = phrase.lower()
                if ph in low and (ph, idx) not in seen:
                    seen.add((ph, idx))
                    found.append((phrase, idx))
        return found

    gap_found = find_exact(student_messages, COMPREHENSION_GAP_PHRASES)

    gap_issues = []
    exact_phrases_collected = []

    for phrase, idx in gap_found:
        phrase_low = phrase.lower()
        msg_low = student_messages[idx].lower()

        has_paraphrase = any(cue in msg_low for cue in PARAPHRASE_CUES)
        has_checkvraag = any(cue in msg_low for cue in CHECKVRAAG_CUES)
        if not has_paraphrase and not has_checkvraag:
            exact_phrases_collected.append(phrase)
            gap_issues.append({
                "phrase": phrase,
                "message_index": idx,
                "exact_message": student_messages[idx],
            })

    understanding_hits = count_hits(UNDERSTANDING_CUES)
    paraphrase_hits = count_hits(PARAPHRASE_CUES)
    checkvraag_hits = count_hits(CHECKVRAAG_CUES)
    confusion_hits = count_hits(CONFUSION_CUES)
    qq = sum(1 for msg in student_messages if "?" in msg)

    reasons = []
    if gap_issues:
        unique = sorted(set(exact_phrases_collected))
        if unique:
            listed = "', '".join(unique[:3])
            if len(unique) > 3:
                listed += f" en {len(unique)-3} anderen"
            reasons.append(
                f"❌ Vermijd uitspraken zoals '{listed}'. Claim geen begrip zonder parafrase of checkvraag."
            )

    if understanding_hits and paraphrase_hits == 0 and checkvraag_hits == 0:
        reasons.append("Je gaf aan dat je het begreep, maar je toetste dit niet met een parafrase of checkvraag.")

    if qq < max(1, understanding_hits):
        reasons.append("Na het uitspreken van begrip volgden weinig verdiepende vragen.")

    if confusion_hits:
        reasons.append("Je sprak onzeker (twijfel / 'uhm' / 'hmm').")

    gap_detected = len(reasons) > 0

    return {
        "gap_detected": gap_detected,
        "reasons": reasons,
        "exact_phrases": sorted(set(exact_phrases_collected)),
        "gap_issues": gap_issues,
        "summary": " ".join(reasons[:3]) if gap_detected else "Je toonde begrip door parafraseren en checkvragen.",
        "student_messages": len(student_messages),
        "understanding_statements": understanding_hits,
        "paraphrase_attempts": paraphrase_hits,
        "checkvraag_attempts": checkvraag_hits,
        "followup_questions": qq,
        "confusion_signals": confusion_hits,
    }
def normalize_pause_distribution(raw_distribution: Optional[Dict[str, float]]) -> Dict[str, float]:
    default = {"short": 0.0, "medium": 0.0, "long": 0.0}
    if not raw_distribution:
        return default

    distances = {k: max(0.0, float(raw_distribution.get(k, 0.0))) for k in default}
    total = sum(distances.values())
    if total <= 0:
        return default

    scale = 100.0 if total <= 1.5 else 1.0
    return {k: round(v * scale if scale == 100.0 else v, 1) for k, v in distances.items()}


def infer_emotion(existing_value: Optional[str], metrics: Dict[str, float]) -> str:
    if existing_value:
        return existing_value.lower()

    tempo_var = metrics.get("tempo_variation", 0)
    filler_ratio = metrics.get("filler_ratio", 0)
    short_pauses = normalize_pause_distribution(metrics.get("pause_distribution", {})).get("short", 0)

    if tempo_var > THRESHOLDS["tempo_variation"]["high"] or filler_ratio > THRESHOLDS["filler_ratio"]["medium"]:
        return "stressed"
    if short_pauses > THRESHOLDS["pause_short_ratio"]:
        return "uncertain"
    if tempo_var < THRESHOLDS["tempo_variation"]["stable"] and filler_ratio < THRESHOLDS["filler_ratio"]["low"]:
        return "calm"
    return "neutral"


def compute_prosody_score(metrics: Dict[str, float], emotion: str) -> float:
    if "prosody_score" in metrics and metrics["prosody_score"] is not None:
        return float(metrics["prosody_score"])

    base = 78.0
    tempo_penalty = max(0, metrics.get("tempo_variation", 0) - THRESHOLDS["tempo_variation"]["stable"]) * 1.2
    filler_penalty = metrics.get("filler_ratio", 0) * 0.6
    pause_avg = metrics.get("average_pause_length", 0)
    pause_penalty = 5 if pause_avg < THRESHOLDS["pause_avg"]["short"] else 0
    if pause_avg > THRESHOLDS["pause_avg"]["long"]:
        pause_penalty += 6

    emotion_adjust = {
        "calm": 4, "empathetic": 6, "neutral": 0,
        "uncertain": -4, "stressed": -6, "confused": -5,
    }

    score = base - tempo_penalty - filler_penalty - pause_penalty + emotion_adjust.get(emotion, 0)
    return max(0, min(100, round(score, 1)))


def build_feedback_metadata(gordon_result: Optional[Dict], speech_result: Optional[Dict]) -> Dict[str, Any]:
    gordon_result = gordon_result or {}
    speech_result = speech_result or {}
    metrics = speech_result.get("metrics", {}) or {}

    # Map avg_pause → average_pause_length if needed
    if "avg_pause" in metrics and "average_pause_length" not in metrics:
        metrics["average_pause_length"] = metrics["avg_pause"]

    pause_distribution = normalize_pause_distribution(metrics.get("pause_distribution"))
    confidence = speech_result.get("confidence", {}) or {}

    # Convert pattern names
    pattern_details = gordon_result.get("pattern_details") or {}

    def resolve(p):
        if isinstance(p, dict):
            return p.get("name") or str(p)
        if isinstance(p, int):
            return pattern_details.get(str(p), {}).get("name", str(p))
        return p

    mentioned_raw = gordon_result.get("mentioned_patterns", []) or []
    missing_raw = gordon_result.get("missing_patterns", []) or []

    mentioned_labels = [resolve(p) for p in mentioned_raw]
    missing_labels = [resolve(p) for p in missing_raw]

    metadata = {
        "speech_rate_wpm": round(metrics.get("speech_rate_wpm", 0)),
        "pause_avg": round(metrics.get("average_pause_length", 0.0), 2),
        "pause_distribution": pause_distribution,
        "tempo_variation": round(metrics.get("tempo_variation", 0), 1),
        "hesitation_markers": int(metrics.get("hesitation_markers", 0)),
        "volume_stability": metrics.get("volume_stability"),
        "filler_ratio": round(metrics.get("filler_ratio", 0), 1),
        "total_words": metrics.get("total_words", 0),
        "confidence_score": confidence.get("score", 0),
        "confidence_level": confidence.get("level", "medium"),
        "confidence_indicators": confidence.get("indicators", []),
        "speech_summary": scrub_text(speech_result.get("summary")),
        "emotion": infer_emotion(speech_result.get("emotion"), metrics),
        "coverage_percentage": round(gordon_result.get("coverage_percentage", 0)),
        "patterns_mentioned": mentioned_labels,
        "patterns_missing": missing_labels,
        "covered_patterns": gordon_result.get("covered_patterns", 0),
        "total_patterns": gordon_result.get("total_patterns", 11),
        "gordon_summary": scrub_text(gordon_result.get("summary")),
    }

    metadata["prosody_score"] = compute_prosody_score(metrics, metadata["emotion"])

    if not metadata["patterns_missing"]:
        default_patterns = [
            "Gezondheidsbeleving","Voeding","Uitscheiding","Activiteit","Slaap",
            "Cognitie","Zelfbeleving","Rollen","Seksualiteit","Stress","Waarden"
        ]
        mentioned_lower = {m.lower() for m in metadata["patterns_mentioned"]}
        metadata["patterns_missing"] = [p for p in default_patterns if p.lower() not in mentioned_lower]

    metadata["llm_prompt"] = build_llm_prompt(metadata)
    return metadata


def build_llm_prompt(metadata: Dict[str, Any]) -> str:
    pause_text = format_pause_distribution_text(metadata["pause_distribution"])
    patterns_mentioned = ", ".join(metadata["patterns_mentioned"]) or "geen"
    patterns_missing = ", ".join(metadata["patterns_missing"]) or "geen"

    prompt = f"""
Je bent een professionele beoordelaar van gespreksvaardigheden.
Gebruik altijd de onderstaande structuur en verwijs naar concrete METRICS.

=====================================
### 1. Complimenten
...

### 2. Communicatiegedrag
...

### 3. Gordon-patronen
...

### 4. Klinische redenering
...

### 5. Concrete vervolgstappen
...

METRICS
Spreeksnelheid: {metadata['speech_rate_wpm']} wpm
Pauze: {metadata['pause_avg']} sec ({pause_text})
Prosodie: {metadata['prosody_score']}
Gordon dekking: {metadata['coverage_percentage']}%
Genoemd: {patterns_mentioned}
Ontbrekend: {patterns_missing}
""".strip()

    return prompt
def build_speech_analysis_section(metadata: Dict[str, Any]) -> str:
    pause_text = format_pause_distribution_text(metadata.get("pause_distribution"))
    return (
        "=== 4. Spraak Analyse ===\n"
        f"- Spreeksnelheid: {metadata['speech_rate_wpm']} woorden/min\n"
        f"- Gemiddelde pauze: {metadata['pause_avg']} sec\n"
        f"- Pauzeverdeling: {pause_text}\n"
        f"- Tempo variatie: {metadata['tempo_variation']}\n"
        f"- Fillerratio: {metadata['filler_ratio']}\n"
        f"- Hesitatie: {metadata['hesitation_markers']}\n"
        f"- Volume stabiliteit: {metadata['volume_stability']}\n"
        f"- Prosodie score: {metadata['prosody_score']}/100\n\n"
        "Spraakobservatie:\n"
        f"{metadata['speech_summary']}\n\n"
    )


# ============================================================
# === 3. Begripstoetsing (OPTION B — Your Detailed Version) ===
# ============================================================

def build_understanding_section(gap_result: Dict[str, Any], conversation_analysis: Dict[str, Any]) -> str:
    section = "=== 3. Begripstoetsing ===\n"

    if not gap_result.get("gap_detected", False):
        return (
            section +
            "Je hebt begrip goed getoetst door parafraseren en verduidelijkende vragen te stellen. "
            "Ga zo door!\n\n"
        )

    section += "We zagen signalen dat je begrip claimde zonder dit actief te controleren.\n\n"

    exact_phrases = gap_result.get("exact_phrases") or []
    if exact_phrases:
        section += "Probleemzinnen:\n"
        for phrase in sorted(set(exact_phrases)):
            section += f"- \"{phrase}\"\n"
        section += "\n"

    if gap_result.get("reasons"):
        section += "Waarom dit belangrijk is:\n"
        for reason in gap_result["reasons"]:
            section += f"- {reason}\n"
        section += "\n"

    section += (
        "Hoe kun je dit verbeteren?\n"
        "- Gebruik parafrases zoals: \"Dus u bedoelt dat…?\"\n"
        "- Voeg checkvragen toe: \"Heb ik dat goed begrepen?\"\n"
        "- Vermijd automatische reacties zoals: \"ik begrijp het\", "
        "\"is goed\", \"precies\", \"I understand\".\n"
        "- Toon actief luisteren door terug te koppelen wat je hoorde.\n\n"
    )

    return section


# ======================================================
# === 5. Gordon-patronen Analyse ===
# ======================================================

def build_gordon_section(metadata: Dict[str, Any]) -> str:
    mentioned = ", ".join(metadata["patterns_mentioned"]) or "geen"
    missing = ", ".join(metadata["patterns_missing"]) or "geen"
    cov = metadata["coverage_percentage"]

    return (
        "=== 5. Gordon-patronen Analyse ===\n"
        f"- Dekking: {cov}%\n"
        f"- Genoemde patronen: {mentioned}\n"
        f"- Ontbrekende patronen: {missing}\n\n"
        "Advies:\n"
        "Probeer ontbrekende patronen te verkennen met gerichte vragen.\n\n"
    )


# ======================================================
# === 6. Actiepunten ===
# ======================================================

def build_action_items_section(
    metadata: Dict[str, Any],
    understanding_result: Dict[str, Any],
    conversation_analysis: Dict[str, Any],
    gordon_result: Optional[Dict[str, Any]] = None,
) -> str:
    section = "=== 6. Actiepunten ===\n"

    # 1. Under­standing / comprehension gap
    if understanding_result.get("gap_detected"):
        section += "- Oefen met parafraseren en checkvragen om begrip te toetsen.\n"

    # 2. Low Gordon coverage
    if conversation_analysis["flags"].get("low_coverage_issue"):
        missing = metadata["patterns_missing"]
        if missing:
            section += f"- Verken ontbrekende Gordon-patronen: {', '.join(missing[:4])}.\n"

    # 3. Filler issue
    if conversation_analysis["flags"].get("filler_issue"):
        fillers = summarize_filler_tokens(conversation_analysis["exact_phrases_used"]["filler_words"])
        section += f"- Verminder stopwoorden (gedetecteerd: {fillers}).\n"

    # 4. Abrupt endings
    if conversation_analysis["flags"].get("abrupt_ending_issue"):
        section += "- Sluit af met een korte samenvatting en duidelijke afspraken.\n"

    if section.strip() == "=== 6. Actiepunten ===":
        section += "Geen directe verbeterpunten — ga zo door!\n\n"
    else:
        section += "\n"

    return section


# ======================================================
# === 7. Afsluiting ===
# ======================================================

def build_motivational_close() -> str:
    return (
        "=== 7. Afsluiting ===\n"
        "Blijf oefenen en toepassen — je groeit zichtbaar in je gespreksvaardigheden.\n"
        "Goed gedaan en ga vooral zo door!\n"
    )


# ======================================================
# === 8. Docentnotities / Instructor Section ===
# ======================================================

def build_lecturer_section(metadata: Dict[str, Any]) -> str:
    return (
        "=== Docentnotitie ===\n"
        "Dit deel is bedoeld voor beoordelaars en omvat ruwe metrics en analyse.\n"
    )


# ======================================================
# === FINAL FEEDBACK ASSEMBLY ===
# ======================================================

def assemble_final_feedback(
    summary_section: str,
    conversation_skills_text: str,
    speech_section_text: str,
    understanding_result: Dict[str, Any],
    conversation_analysis: Dict[str, Any],
    metadata: Dict[str, Any],
    gordon_result: Optional[Dict[str, Any]] = None,
) -> str:

    understanding_section_text = build_understanding_section(
        understanding_result,
        conversation_analysis
    )

    action_items_text = build_action_items_section(
        metadata,
        understanding_result,
        conversation_analysis,
        gordon_result,
    )

    gordon_section_text = build_gordon_section(metadata)
    motivational_close = build_motivational_close()
    lecturer_section_text = build_lecturer_section(metadata)

    ordered_sections = [
        summary_section,
        conversation_skills_text,
        understanding_section_text,   # === 3. Begripstoetsing ===
        speech_section_text,          # === 4. Spraak Analyse ===
        gordon_section_text,          # === 5. Gordon ===
        action_items_text,            # === 6. Actiepunten ===
        motivational_close,           # === 7. Afsluiting ===
        lecturer_section_text,        # === Docentnotitie ===
    ]

    return "\n".join(ordered_sections)
