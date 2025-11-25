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
COMPREHENSION_GAP_PHRASES = [
    "ik begrijp het",
    "ik snap het",
    "oké, duidelijk",
    "oke, duidelijk",
    "we gaan het regelen",
    "ik begrijp u",
    "ja, ik snap het",
    "ja ik snap het",
    "ik weet het",
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
]

# Heuristics to spot (perceived) understanding, paraphrasing, and confusion.
UNDERSTANDING_CUES = [
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
]
PARAPHRASE_CUES = [
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
]
CHECKVRAAG_CUES = [
    "heb ik dat goed",
    "klopt dat",
    "begrijp ik u goed",
    "heb ik het goed begrepen",
    "is dat correct",
    "is dat juist",
    "vraag",
    "?",
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
        """Find exact phrases used and their message index."""
        found = []
        for idx, msg in enumerate(messages):
            msg_lower = msg.lower()
            for phrase in phrases:
                if phrase in msg_lower:
                    # Extract the exact phrase as it appears (case-insensitive match)
                    found.append((phrase, idx))
        return found

    # Find comprehension gap phrases
    gap_phrases_found = find_exact_phrases(student_messages, COMPREHENSION_GAP_PHRASES)
    
    # Check if each gap phrase is followed by paraphrase or checkvraag
    gap_issues = []
    exact_phrases_quoted = []
    
    for phrase, msg_idx in gap_phrases_found:
        # Check if paraphrase or checkvraag appears in the SAME message or in next 1-2 messages
        followed_by_evidence = False
        
        # First check the same message (after the phrase)
        current_message = student_messages[msg_idx].lower()
        phrase_pos = current_message.find(phrase.lower())
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
            gap_reasons.append(f"❌ VERMIJD dit: Je zei '{phrases_text}' zonder te toetsen of je het goed begreep. Dit veroorzaakt een begripsgat. Zeg NIET 'ik begrijp het' of 'ja dat klopt' zonder direct daarna te controleren met een parafrase of checkvraag.")
        else:
            gap_reasons.append(f"❌ VERMIJD dit: Je zei meerdere keren dingen zoals '{phrases_text}' ({total_count} keer) zonder te toetsen of je het goed begreep. Dit veroorzaakt een begripsgat. Zeg NIET 'ik begrijp het', 'ja ik snap het', 'ja dat klopt' of vergelijkbare uitspraken zonder direct daarna te controleren met een parafrase (bijv. 'Dus u voelt zich...?') of checkvraag (bijv. 'Heb ik dat goed begrepen?').")

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


def build_summary_section(metadata: Dict[str, Any], conversation_history: Optional[str] = None) -> str:
    """
    Create the top summary with badges and quick metrics.
    Rule 4: If coverage < 3/11 → require strong negative feedback (orange/red).
    Rule 5: If history < 5 sentences → force warning.
    """
    lines = ["=== Samenvatting ==="]

    # Overall verdict based on coverage, prosody, and comprehension check.
    coverage = metadata["coverage_percentage"]
    covered_patterns = metadata.get("covered_patterns", 0)
    prosody = metadata["prosody_score"]
    gap_result = metadata.get("understanding_gap") or {}
    has_gap = gap_result.get("gap_detected")

    # Rule 4: Low coverage handling (< 3/11 patterns = ~27%)
    coverage_ratio = covered_patterns / 11 if metadata.get("total_patterns", 11) > 0 else 0
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
        if has_gap:
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
    if filler_ratio <= THRESHOLDS["filler_ratio"]["low"]:
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

    gap_result = metadata.get("understanding_gap") or {}
    if gap_result.get("student_messages", 0) == 0:
        lines.append("- ℹ️ Begripscontrole: geen uitspraken om begrip te toetsen.")
    elif gap_result.get("gap_detected"):
        # Rule 3: Quote exact phrases and give explicit directive
        exact_phrases = gap_result.get("exact_phrases", [])
        if exact_phrases:
            phrases_quoted = "', '".join(set(exact_phrases[:3]))
            lines.append(f"- ❌ Begripscontrole: Je gebruikte uitspraken zoals '{phrases_quoted}' zonder te toetsen. VERMIJD dit - zeg NIETS of gebruik direct een parafrase/checkvraag.")
        else:
            lines.append(f"- ❌ Begripscontrole: {gap_result.get('summary', 'Begrip niet overtuigend getoond.')}")
    else:
        lines.append("- ✅ Begripscontrole: parafrases en vervolgvragen maakten je begrip overtuigend.")

    # Rule 4: List missing patterns for low coverage
    missing_patterns = metadata.get("patterns_missing") or []
    if is_low_coverage and missing_patterns:
        top_missing = ", ".join(missing_patterns[:4])
        lines.append(f"- ❌ Ontbrekende patronen: {top_missing} (essentieel voor veilige anamnese)")
    elif missing_patterns:
        top_missing = ", ".join(missing_patterns[:2])
        lines.append(f"- Volgende focus: vraag door op {top_missing} met concrete voorbeelden.")

    lines.append(
        f"- Metrics: {metadata['speech_rate_wpm']} wpm | tempo-variatie {metadata['tempo_variation']}% | "
        f"pauze {metadata['pause_avg']}s | prosodie {metadata['prosody_score']}/100 | emotie {metadata['emotion']}."
    )
    return "\n".join(lines)


def format_pause_distribution_text(pause_distribution: Dict[str, float]) -> str:
    return f"{pause_distribution['short']}% kort / {pause_distribution['medium']}% middel / {pause_distribution['long']}% lang"


def build_speech_section(speech_result: Optional[Dict], metadata: Dict[str, Any]) -> Optional[str]:
    """
    Detailed speech analytics with numeric metrics.
    Rule 2: MUST include filler detection and density per 10 words.
    Rule 6: Include speech rate feedback.
    """
    if not speech_result:
        return None

    pause_text = format_pause_distribution_text(metadata["pause_distribution"])
    lines = ["=== Spraak Analyse ===", "**Belangrijkste metingen**"]
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
    filler_count = metadata.get('hesitation_markers', 0)
    filler_ratio = metadata.get('filler_ratio', 0)
    total_words = metadata.get('total_words', 0) or speech_result.get('metrics', {}).get('total_words', 0)
    
    if filler_count > 0:
        # Calculate density per 10 words
        filler_density = (filler_count / total_words * 10) if total_words > 0 else 0
        lines.append(f"- Opvulgeluidjes/fillers: {filler_count} keer gebruikt")
        lines.append(f"- Filler-dichtheid: {filler_density:.1f} per 10 woorden")
        lines.append("  → Opvulgeluidjes verminderen de duidelijkheid van je communicatie.")
        
        # Rule 2C: Severity indicator if > 3
        if filler_count > 3:
            lines.append(f"  → Je gebruikte veel opvulgeluidjes ({filler_count} keer), wat de professionaliteit verlaagt.")
    else:
        lines.append("- Opvulgeluidjes/fillers: geen gedetecteerd")
    
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


def build_understanding_section(gap_result: Dict[str, Any]) -> str:
    """
    Summarize gaps between uitgesproken en aangetoond begrip.
    Rule 1: Must quote exact phrases that caused issues.
    Rule 3: Always quote exact student phrases.
    """
    lines = ["=== Begripstoetsing ==="]
    if not gap_result or gap_result.get("student_messages", 0) == 0:
        lines.append("- Geen studentuitspraken beschikbaar om begrip te toetsen.")
        return "\n".join(lines)

    lines.append(
        f"- Uitgesproken begrip: {gap_result.get('understanding_statements', 0)} | parafrases: "
        f"{gap_result.get('paraphrase_attempts', 0)} | checkvragen: {gap_result.get('checkvraag_attempts', 0)} | vervolgvragen: {gap_result.get('followup_questions', 0)}."
    )

    if gap_result.get("gap_detected"):
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
                lines.append(f"- ❌ VERMIJD: Je zei '{phrases_text}' zonder te toetsen of je het goed begreep. Zeg NIET dingen zoals 'ik begrijp het', 'ja ik snap het', 'ja dat klopt' of vergelijkbare uitspraken zonder direct daarna een parafrase ('Dus u zegt dat...?') of checkvraag ('Heb ik dat goed begrepen?') te stellen.")
            else:
                lines.append(f"- ❌ VERMIJD: Je gebruikte {total_count} keer uitspraken zoals '{phrases_text}' zonder te toetsen of je het goed begreep. Dit veroorzaakt een begripsgat. Zeg NIET 'ik begrijp het', 'ja ik snap het', 'ja dat klopt', 'precies', 'akkoord' of vergelijkbare bevestigingen zonder direct daarna te controleren met een parafrase ('Dus u voelt zich...?') of checkvraag ('Klopt het dat...?').")
        
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
    lines = ["=== Gordon Patronen Analyse ==="]
    coverage = metadata["coverage_percentage"]
    covered_patterns = metadata['covered_patterns']
    total_patterns = metadata['total_patterns']
    
    lines.append(f"- Dekking: {covered_patterns}/{total_patterns} ({coverage:.1f}%)")

    # Rule 4: Strong feedback for low coverage
    is_low_coverage = covered_patterns < 3 or coverage < 27.3
    if is_low_coverage:
        lines.append(f"- ⚠️ Lage dekking: Door slechts {covered_patterns} patronen te behandelen, mis je belangrijke informatie die essentieel is voor een veilige anamnese.")

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
    strengths: List[str] = []
    improvements: List[str] = []
    techniques: List[str] = []
    
    filler_ratio = metadata["filler_ratio"]
    filler_count = metadata.get("hesitation_markers", 0)
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
    if emotion in {"uncertain", "stressed"}:
        improvements.append("Je klonk wat onzeker; vertraag je ademhaling en vat antwoorden samen om vertrouwen te tonen.")
    if prosody < THRESHOLDS["prosody"]["ok"]:
        improvements.append("Werk aan vocale variatie door sleutelwoorden te benadrukken en toonhoogte licht te variëren.")
    if volume_stability and volume_stability < THRESHOLDS["volume_stability"]:
        improvements.append("Houd je volume stabiel door rechtop te zitten en uit te ademen tijdens het spreken.")
    if coverage < 60:
        missing = ", ".join(metadata["patterns_missing"][:3])
        improvements.append(f"Plan vragen rond ontbrekende patronen ({missing}) om vollediger te screenen.")
    if gap_result.get("gap_detected"):
        exact_phrases = gap_result.get("exact_phrases", [])
        if exact_phrases:
            phrases_example = "', '".join(exact_phrases[:3])
            improvements.append(f"VERMIJD uitspraken zoals '{phrases_example}' zonder te controleren. In plaats daarvan: zeg NIETS, of gebruik direct een parafrase ('Dus u zegt dat...?') of checkvraag ('Heb ik dat goed begrepen?').")
        else:
            improvements.append("VERMIJD uitspraken zoals 'ik begrijp het', 'ja ik snap het', 'ja dat klopt' zonder te controleren. In plaats daarvan: gebruik direct een parafrase ('Dus u zegt dat...?') of checkvraag ('Heb ik dat goed begrepen?').")

    # Collect specific communication techniques (Rule 7)
    if gap_result.get("gap_detected"):
        exact_phrases = gap_result.get("exact_phrases", [])
        if exact_phrases:
            phrases_example = "', '".join(exact_phrases[:2])
            techniques.append(f"Techniek: VERMIJD uitspraken zoals '{phrases_example}'. In plaats daarvan: zeg NIETS, of gebruik direct een parafrase ('Dus u voelt zich...?') of checkvraag ('Heb ik dat goed begrepen?') om je begrip te toetsen.")
        else:
            techniques.append("Techniek: VERMIJD uitspraken zoals 'ik begrijp het', 'ja ik snap het', 'ja dat klopt'. In plaats daarvan: zeg NIETS, of gebruik direct een parafrase ('Dus u voelt zich...?') of checkvraag ('Heb ik dat goed begrepen?') om je begrip te toetsen.")
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

    lines = ["=== Actiepunten ==="]
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
        return "=== Afsluiting ===\n\nJe hebt een solide basis gelegd. Blijf oefenen met de actiepunten en je zult nog sterker worden in je gespreksvaardigheden. Succes met de volgende oefening!"
    elif coverage >= 40:
        return "=== Afsluiting ===\n\nJe maakt goede vooruitgang. Focus op de verbeterpunten en blijf vooral veel oefenen. Elke oefening maakt je beter!"
    else:
        return "=== Afsluiting ===\n\nDit is een leerproces. Pak de actiepunten op en probeer het opnieuw. Met oefening wordt je steeds beter in het voeren van een goede anamnese."


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

    # Detect gaps tussen uitgesproken en getoond begrip.
    gap_result = analyze_understanding_gaps(conversation_history, metadata)
    metadata["understanding_gap"] = gap_result

    summary_section = build_summary_section(metadata, conversation_history)

    llm_sections, lecturer_notes = sanitize_llm_output(conversation_feedback, metadata)
    llm_block = ["=== Gespreksvaardigheden (LLM) ==="]
    for header in LLM_SECTION_TITLES:
        llm_block.append(f"**{header}**")
        llm_block.append(llm_sections[header])
    llm_section_text = "\n\n".join(llm_block)

    speech_section_text = build_speech_section(speech_result, metadata)
    understanding_section_text = build_understanding_section(gap_result)
    gordon_section_text = build_gordon_section(metadata)
    action_items_text = build_action_items(metadata)
    lecturer_section_text = build_lecturer_notes_section(lecturer_notes)

    # Rule 8: Add motivational close
    motivational_close = build_motivational_close(metadata)
    
    ordered_sections: List[str] = [
        summary_section,
        llm_section_text,
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
        "llm": {
            "title": "=== Gespreksvaardigheden (LLM) ===",
            "sections": llm_sections,
        },
        "understanding_gap": understanding_section_text,
        "gordon": gordon_section_text,
        "action_items": action_items_text,
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
