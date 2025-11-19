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

    return sections, scrub_text(lecturer_notes)


def build_summary_section(metadata: Dict[str, Any]) -> str:
    """
    Create the top summary with badges and quick metrics.
    """
    lines = ["=== Samenvatting ==="]
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
    """
    if not speech_result:
        return None

    pause_text = format_pause_distribution_text(metadata["pause_distribution"])
    lines = ["=== Spraak Analyse ===", "**Belangrijkste metingen**"]
    lines.append(f"- Spreeksnelheid: {metadata['speech_rate_wpm']} wpm")
    lines.append(f"- Tempo-variatie: {metadata['tempo_variation']}%")
    lines.append(f"- Pauzedistributie: {pause_text}")
    lines.append(f"- Gemiddelde pauzeduur: {metadata['pause_avg']} s")
    lines.append(f"- Hesitatie-markers: {metadata['hesitation_markers']}")
    lines.append(f"- Volume-stabiliteit: {metadata['volume_stability'] or 'n.v.t.'}")
    lines.append(f"- Stopwoorden: {metadata['filler_ratio']}%")
    lines.append(f"- Prosodie: {metadata['prosody_score']}/100")
    lines.append(f"- Gevoelstoon: {metadata['emotion']}")

    summary = metadata["speech_summary"]
    if summary:
        lines.append("\n**Interpretatie**")
        lines.append(summary)

    indicators = metadata.get("confidence_indicators") or []
    if indicators:
        lines.append("\n**Indicatoren zelfvertrouwen**")
        lines.extend([f"- {indicator}" for indicator in indicators[:5]])

    return "\n".join(lines)


def build_gordon_section(metadata: Dict[str, Any]) -> str:
    """
    Describe Gordon pattern coverage and missing aspects.
    """
    lines = ["=== Gordon Patronen Analyse ==="]
    lines.append(f"- Dekking: {metadata['covered_patterns']}/{metadata['total_patterns']} ({metadata['coverage_percentage']}%)")

    mentioned = ", ".join(metadata["patterns_mentioned"]) if metadata["patterns_mentioned"] else "geen vermeld"
    missing = ", ".join(metadata["patterns_missing"]) if metadata["patterns_missing"] else "n.v.t."
    lines.append(f"- Genoemde patronen: {mentioned}")
    lines.append(f"- Ontbrekende patronen: {missing}")

    if metadata["gordon_summary"]:
        lines.append(f"- Samenvatting: {metadata['gordon_summary']}")

    top_missing = ", ".join(metadata["patterns_missing"][:3]) if metadata["patterns_missing"] else "n.v.t."
    lines.append(f"- Focus voor volgende keer: {top_missing}")
    return "\n".join(lines)


def build_action_items(metadata: Dict[str, Any]) -> str:
    """
    Use speech + Gordon metrics to craft concrete action steps.
    """
    actions: List[str] = []
    filler_ratio = metadata["filler_ratio"]
    tempo_variation = metadata["tempo_variation"]
    speech_rate = metadata["speech_rate_wpm"]
    pause_distribution = metadata["pause_distribution"]
    pause_avg = metadata["pause_avg"]
    emotion = metadata["emotion"]
    hesitations = metadata["hesitation_markers"]
    prosody = metadata["prosody_score"]
    volume_stability = metadata["volume_stability"]

    if tempo_variation > THRESHOLDS["tempo_variation"]["high"]:
        actions.append("Probeer rustiger en consistenter te spreken; stabiliseer je tempo per vraag.")
    if speech_rate < THRESHOLDS["speech_rate"]["slow"]:
        actions.append("Verhoog je spreeksnelheid licht zodat het gesprek levendiger blijft (doel 105-125 wpm).")
    elif speech_rate > THRESHOLDS["speech_rate"]["fast"]:
        actions.append("Vertraag bewust door na elke vraag een korte pauze te nemen voor helderheid.")
    if pause_distribution["short"] > THRESHOLDS["pause_short_ratio"] or pause_avg < THRESHOLDS["pause_avg"]["short"]:
        actions.append("Kort en veel pauzes kunnen onzekerheid tonen; adem diep uit voordat je reageert.")
    if filler_ratio > THRESHOLDS["filler_ratio"]["medium"]:
        actions.append("Noteer steekwoorden vooraf om 'euh' of 'uh' te beperken en meer rust te behouden.")
    if emotion in {"uncertain", "stressed"}:
        actions.append("Je klonk wat onzeker; vertraag je ademhaling en vat antwoorden samen om vertrouwen te tonen.")
    if hesitations > THRESHOLDS["hesitation_markers"]["some"]:
        actions.append("Oefen met stiltes te laten vallen in plaats van hesitatie-geluiden wanneer je nadenkt.")
    if prosody < THRESHOLDS["prosody"]["ok"]:
        actions.append("Werk aan vocale variatie door sleutelwoorden te benadrukken en toonhoogte licht te variëren.")
    if volume_stability and volume_stability < THRESHOLDS["volume_stability"]:
        actions.append("Houd je volume stabiel door rechtop te zitten en uit te ademen tijdens het spreken.")
    if metadata["coverage_percentage"] < 60:
        missing = ", ".join(metadata["patterns_missing"][:3])
        actions.append(f"Plan vragen rond ontbrekende patronen ({missing}) om vollediger te screenen.")

    actions = list(dict.fromkeys(actions))  # Deduplicate preserving order
    if len(actions) < 3:
        fallback_actions = [
            "Gebruik een checklijst met Gordon patronen om structuur te houden.",
            "Herhaal de kernwoorden van de patiënt om empathie te bevestigen.",
            "Noteer tijdens het gesprek kort wat al besproken is om vervolgvragen beter te plannen.",
        ]
        for suggestion in fallback_actions:
            if suggestion not in actions:
                actions.append(suggestion)
            if len(actions) >= 3:
                break

    actions = actions[:6]
    lines = ["=== Actiepunten ==="]
    for action in actions:
        lines.append(f"- {action}")
    return "\n".join(lines)


def build_lecturer_notes_section(notes: Optional[str]) -> Optional[str]:
    if not notes:
        return None
    return "=== Optionele Docentnotities ===\n\n*" + scrub_text(notes) + "*"


def format_student_feedback(conversation_feedback, gordon_result, speech_result):
    """
    Format all feedback components into a clear, student-friendly structure.
    Returns a dict so routes can pass only the plain text to TTS while still exposing
    structured metadata to the client.
    """
    metadata = build_feedback_metadata(gordon_result, speech_result)
    llm_prompt = metadata.get("llm_prompt") or build_llm_prompt(metadata)
    metadata["llm_prompt"] = llm_prompt

    summary_section = build_summary_section(metadata)

    llm_sections, lecturer_notes = sanitize_llm_output(conversation_feedback, metadata)
    llm_block = ["=== Gespreksvaardigheden (LLM) ==="]
    for header in LLM_SECTION_TITLES:
        llm_block.append(f"**{header}**")
        llm_block.append(llm_sections[header])
    llm_section_text = "\n\n".join(llm_block)

    speech_section_text = build_speech_section(speech_result, metadata)
    gordon_section_text = build_gordon_section(metadata)
    action_items_text = build_action_items(metadata)
    lecturer_section_text = build_lecturer_notes_section(lecturer_notes)

    ordered_sections: List[str] = [
        summary_section,
        llm_section_text,
        speech_section_text,
        gordon_section_text,
        action_items_text,
        lecturer_section_text,
    ]

    formatted_feedback = "\n\n".join(filter(None, ordered_sections))

    structured_sections: Dict[str, Any] = {
        "summary": summary_section,
        "llm": {
            "title": "=== Gespreksvaardigheden (LLM) ===",
            "sections": llm_sections,
        },
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
