"""
Deterministic phase-based conversation detector for nursing feedback.
Implements rubric-driven scoring for phases 1–3 plus Gordon pattern coverage.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# -----------------
# Configuration
# -----------------

DEFAULT_PHASE_CONFIG: Dict[str, Any] = {
    "phase1_time_window": (60, 90),  # seconds
    "phase3_time_window": (45, 60),  # seconds
    "phase1_turn_pct": (0.10, 0.15),
    "phase3_turn_pct": (0.10, 0.15),
    "phase_min_turns": 2,
    "paraphrase_overlap": 0.25,
    "e_in_overlap": 0.1,
    "e_ex_overlap_low": 0.05,
    "active_listening_cap": 5,
    "stopwords": {
        "ik",
        "je",
        "jij",
        "u",
        "we",
        "wij",
        "ze",
        "zij",
        "hij",
        "zij",
        "het",
        "een",
        "de",
        "het",
        "en",
        "of",
        "maar",
        "dat",
        "dit",
        "er",
        "te",
        "van",
        "voor",
        "naar",
        "met",
        "op",
        "in",
        "is",
        "ben",
        "was",
        "zijn",
        "hebt",
        "heb",
        "heeft",
        "had",
        "niet",
        "geen",
        "wel",
        "dan",
        "als",
        "ook",
        "dus",
        "om",
        "tot",
        "deze",
        "die",
        "dit",
        "hier",
        "daar",
        "een",
        "the",
        "and",
        "or",
        "but",
        "a",
        "an",
        "to",
    },
}

PHASE1_ITEMS = [
    "greeting_intro",
    "align_goal_content_time",
    "small_talk_comfort",
    "needs_expectations",
]

PHASE2_ITEMS = [
    "open_questions",
    "closed_questions",
    "active_listening",
    "understanding_checks",
    "interim_summaries",
    "paraphrase_repeat_to_continue",
    "e_in_deepen",
    "e_ex_shift_topic",
    "content_and_emotional_summary",
]

PHASE3_ITEMS = [
    "end_summary",
    "follow_up_agreements",
    "professional_closing",
    "encourage_participation",
]

PHASE_LABELS = {
    "phase1": "Fase 1 - Contact maken",
    "phase2": "Fase 2 - Exploreren (anamnese)",
    "phase3": "Fase 3 - Afronding",
}

ITEM_LABELS = {
    "greeting_intro": "Begroeting + introductie",
    "align_goal_content_time": "Doel/informatie (+ tijd) afstemmen",
    "small_talk_comfort": "Small talk / comfort",
    "needs_expectations": "Behoeften en verwachtingen peilen",
    "open_questions": "Open vragen (exploreren)",
    "closed_questions": "Gesloten vragen (feiten)",
    "active_listening": "Actief luisteren / backchannels",
    "understanding_checks": "Begrip toetsen",
    "interim_summaries": "Tussentijdse samenvattingen",
    "paraphrase_repeat_to_continue": "Herhalen/parafraseren om door te laten gaan",
    "e_in_deepen": "E-in vragen (verdiepen zelfde onderwerp)",
    "e_ex_shift_topic": "E-ex vragen (onderwerp wisselen)",
    "content_and_emotional_summary": "Inhoud + gevoel samenvatten",
    "end_summary": "Eindsamenvatting",
    "follow_up_agreements": "Concrete vervolgafspraken",
    "professional_closing": "Professionele afronding",
    "encourage_participation": "Patient betrekken bij oplossingen",
}

SUMMARY_FRAMING_CUES = [
    "even samenvatten",
    "dus tot nu toe",
    "als ik het samenvat",
    "kort gezegd",
    "samenvattend",
    "we hebben besproken",
    "dus u zegt",
]
SUMMARY_REFLECTIVE_CUES = [
    "u zegt",
    "u geeft aan",
    "u vertelde",
    "u benoemt",
    "u ervaart",
    "u heeft",
    "u bent",
]

EMOTION_CUES = [
    # Reflection framings rather than raw emotion words
    "ik hoor dat",
    "dat klinkt",
    "dat lijkt me",
    "u voelt zich",
    "u maakt zich zorgen",
]

GORDON_PATTERNS: Dict[str, Dict[str, Any]] = {
    "1": {
        "name": "Health Perception–Health Management",
        "keywords": [
            "gezondheid",
            "klacht",
            "klachten",
            "diagnose",
            "medicatie",
            "therapie",
            "huisarts",
            "ziekenhuis",
            "allergie",
            "roken",
            "alcohol",
            "vaccin",
            "zelfzorg",
            "behandeling",
            "zorg",
        ],
    },
    "2": {
        "name": "Nutritional–Metabolic",
        "keywords": [
            "eet",
            "eten",
            "drink",
            "drinken",
            "appetijt",
            "misselijk",
            "gewicht",
            "afvallen",
            "aankomen",
            "dieet",
            "voeding",
            "slikken",
            "dorst",
            "diabetes",
        ],
    },
    "3": {
        "name": "Elimination",
        "keywords": [
            "plassen",
            "urine",
            "ontlasting",
            "diarree",
            "obstipatie",
            "stoelgang",
            "incontinentie",
        ],
    },
    "4": {
        "name": "Activity–Exercise",
        "keywords": [
            "lopen",
            "trap",
            "sport",
            "bewegen",
            "conditie",
            "benauwd",
            "kortademig",
            "mobiliteit",
            "adl",
            "activiteiten",
        ],
    },
    "5": {
        "name": "Sleep–Rest",
        "keywords": [
            "slapen",
            "inslapen",
            "doorslapen",
            "wakker",
            "nacht",
            "rust",
            "vermoeid",
            "slaapritme",
        ],
    },
    "6": {
        "name": "Cognitive–Perceptual",
        "keywords": [
            "pijn",
            "pijnscore",
            "duizelig",
            "gehoor",
            "zien",
            "tinteling",
            "verward",
            "concentratie",
            "geheugen",
        ],
    },
    "7": {
        "name": "Self-Perception–Self-Concept",
        "keywords": [
            "zelfbeeld",
            "schaam",
            "trots",
            "onzeker",
            "zekerheid",
            "lichaamsbeeld",
            "ik voel me",
        ],
    },
    "8": {
        "name": "Roles–Relationships",
        "keywords": [
            "familie",
            "partner",
            "kinderen",
            "werk",
            "mantelzorg",
            "steun",
            "relatie",
            "thuis",
            "sociaal",
        ],
    },
    "9": {
        "name": "Sexuality–Reproductive",
        "keywords": [
            "seks",
            "libido",
            "menstruatie",
            "zwanger",
            "anticonceptie",
            "intimiteit",
            "erectie",
            "vruchtbaarheid",
        ],
    },
    "10": {
        "name": "Coping–Stress Tolerance",
        "keywords": [
            "stress",
            "spanning",
            "angst",
            "somber",
            "coping",
            "omgaan met",
            "paniek",
            "zorgen",
            "overbelast",
        ],
    },
    "11": {
        "name": "Values–Beliefs",
        "keywords": [
            "geloof",
            "religie",
            "spiritualiteit",
            "waarden",
            "overtuiging",
            "cultuur",
            "belangrijk",
        ],
    },
}

# -----------------
# Helpers
# -----------------


def normalize_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"[^\w\s\?]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def content_words(text: str, stopwords: Set[str]) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in stopwords]


def jaccard_overlap(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    inter = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(inter) / len(union)


def _init_phase_structure() -> Dict[str, Any]:
    phases: Dict[str, Any] = {
        "phase1": {"score_total": 0, "max": len(PHASE1_ITEMS) * 2, "items": {}},
        "phase2": {"score_total": 0, "max": len(PHASE2_ITEMS) * 2, "items": {}},
        "phase3": {"score_total": 0, "max": len(PHASE3_ITEMS) * 2, "items": {}},
    }
    for key in PHASE1_ITEMS:
        phases["phase1"]["items"][key] = {"present": False, "score": 0, "evidence": []}
    for key in PHASE2_ITEMS:
        phases["phase2"]["items"][key] = {"present": False, "score": 0, "evidence": []}
    for key in PHASE3_ITEMS:
        phases["phase3"]["items"][key] = {"present": False, "score": 0, "evidence": []}
    return phases


def _empty_metrics() -> Dict[str, Any]:
    return {
        "open_q_count": 0,
        "closed_q_count": 0,
        "active_listening_count": 0,
        "understanding_check_count": 0,
        "interim_summary_count": 0,
        "paraphrase_count": 0,
        "e_in_count": 0,
        "e_ex_count": 0,
        "emotional_reflection_count": 0,
        "gordon_covered_count": 0,
        "gordon_coverage_percent": 0.0,
    }


def _segment_by_time(turns: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[int, str]:
    timestamps = [t.get("timestamp") for t in turns if t.get("timestamp") is not None]
    if not timestamps:
        return {}
    start = min(timestamps)
    end = max(timestamps)
    phase1_end = start + config["phase1_time_window"][1]
    phase3_start = end - config["phase3_time_window"][1]
    if phase3_start <= phase1_end:
        # Conversation too short: fall back to percentage split
        return {}
    mapping: Dict[int, str] = {}
    for idx, turn in enumerate(turns):
        ts = turn.get("timestamp")
        if ts is None:
            continue
        if ts <= phase1_end:
            mapping[idx] = "phase1"
        elif ts >= phase3_start:
            mapping[idx] = "phase3"
        else:
            mapping[idx] = "phase2"
    return mapping


def _segment_by_turns(turns: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[int, str]:
    """
    Fallback segmentation when timestamps are absent.
    Assign phases over absolute turn indices (not only student turns) so patient turns stay aligned.
    """
    mapping: Dict[int, str] = {}
    total_turns = len(turns)
    if total_turns == 0:
        return mapping
    if total_turns <= 3:
        n_phase1 = 1
        n_phase3 = 0
    elif total_turns <= 4:
        n_phase1 = 1
        n_phase3 = 1
    else:
        n_phase1 = max(
            config["phase_min_turns"],
            int(math.ceil(total_turns * config["phase1_turn_pct"][1])),
        )
        n_phase3 = max(
            config["phase_min_turns"],
            int(math.ceil(total_turns * config["phase3_turn_pct"][1])),
        )
        if n_phase1 + n_phase3 > total_turns:
            n_phase3 = max(1, total_turns - n_phase1)

    phase1_end = n_phase1 - 1
    phase3_start = total_turns - n_phase3

    for idx in range(total_turns):
        if idx <= phase1_end:
            mapping[idx] = "phase1"
        elif idx >= phase3_start:
            mapping[idx] = "phase3"
        else:
            mapping[idx] = "phase2"
    return mapping


def segment_phases(turns: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[int, str]:
    if not turns:
        return {}
    by_time = _segment_by_time(turns, config)
    if by_time:
        return by_time
    return _segment_by_turns(turns, config)


def _add_evidence(
    phases: Dict[str, Any],
    phase_key: str,
    item_key: str,
    turn_index: int,
    quote: str,
    timestamp: Optional[float],
) -> None:
    item = phases[phase_key]["items"][item_key]
    if len(item["evidence"]) < 2:
        item["evidence"].append(
            {
                "turn": turn_index,
                "timestamp": timestamp,
                "quote": quote.strip(),
            }
        )


def _add_global_evidence(
    evidence: List[Dict[str, Any]],
    phase_key: str,
    item_key: str,
    turn_index: int,
    timestamp: Optional[float],
    quote: str,
    speaker: str,
) -> None:
    evidence.append(
        {
            "phase": phase_key,
            "item": item_key,
            "turn": turn_index,
            "timestamp": timestamp,
            "quote": quote.strip(),
            "speaker": speaker,
        }
    )


def _record_hit(
    hit_map: Dict[str, Dict[str, Dict[str, int]]], phase: str, item: str, strength: str
) -> None:
    phase_hits = hit_map.setdefault(phase, {})
    item_hits = phase_hits.setdefault(item, {"strong": 0, "weak": 0})
    item_hits[strength] += 1


def _finalize_scores(
    phases: Dict[str, Any], hit_map: Dict[str, Dict[str, Dict[str, int]]]
) -> None:
    for phase_key, phase_data in phases.items():
        total = 0
        for item_key, item_data in phase_data["items"].items():
            hits = hit_map.get(phase_key, {}).get(item_key, {"strong": 0, "weak": 0})
            score = 0
            if hits["strong"] > 0:
                score = 2
            elif hits["weak"] > 0:
                score = 1
            item_data["score"] = score
            item_data["present"] = score > 0
            total += score
        phase_data["score_total"] = total


# -----------------
# Detection helpers
# -----------------


def _looks_like_question(text: str) -> bool:
    """
    Lenient question detector for STT: do not require punctuation.
    """
    t = text.strip().lower()
    if "?" in t:
        return True
    starters = (
        "wat",
        "hoe",
        "waar",
        "wanneer",
        "waarom",
        "waardoor",
        "heeft u",
        "bent u",
        "neemt u",
        "is het",
        "doet het",
        "kan",
        "kunt u",
        "zou u",
        "mag ik vragen",
        "sinds wanneer",
        "hoe lang",
        "hoeveel",
        "hoe vaak",
        "welke",
    )
    return any(t.startswith(s) for s in starters)


# Public alias for other modules (formatter) to reuse question heuristic.
def looks_like_question(text: str) -> bool:
    return _looks_like_question(text)


def _is_open_question(text: str) -> bool:
    """
    Detect open questions without requiring '?' (robust for STT).
    """
    t = text.strip().lower()
    open_starts = (
        "kunt u vertellen",
        "zou u kunnen vertellen",
        "wat maakt dat",
        "kunt u een",
        "kunt u me",
        "kunt u mij",
        "hoe",
        "wat",
        "waar",
        "wanneer",
        "waarom",
        "waardoor",
        "welke",
    )
    return t.startswith(open_starts)


def _is_closed_question(text: str) -> bool:
    """
    Detect closed/yes-no questions without requiring '?'.
    """
    t = text.strip().lower()
    yesno_starts = ("heeft u", "bent u", "neemt u", "is het", "doet het", "mag ik vragen", "kan", "kunt u")
    factual = ("hoeveel", "sinds wanneer", "hoe lang", "welke medicatie", "hoe vaak")
    if any(t.startswith(s) for s in yesno_starts):
        return True
    return any(t.startswith(cue) for cue in factual)


def _is_active_listening(text: str) -> bool:
    """
    Count only brief backchannels/acknowledgements; never count questions.
    """
    if _looks_like_question(text):
        return False
    backchannels = {
        "hm",
        "hmm",
        "mhm",
        "ja",
        "ok",
        "oké",
        "okee",
        "begrijpelijk",
        "dank u",
        "ga verder",
        "vertelt u verder",
    }
    cleaned = text.strip().lower()
    return cleaned in backchannels


def _has_understanding_check(text: str) -> bool:
    cues = [
        "klopt dat",
        "begrijp ik goed",
        "als ik u goed begrijp",
        "heb ik u goed begrepen",
    ]
    return any(cue in text for cue in cues)


def _has_interim_summary(text: str) -> bool:
    framing = any(cue in text for cue in SUMMARY_FRAMING_CUES)
    reflective = any(cue in text for cue in SUMMARY_REFLECTIVE_CUES)
    return framing and reflective


def _has_emotional_reflection(text: str) -> bool:
    return any(cue in text for cue in EMOTION_CUES)


def _has_content_summary(text: str) -> bool:
    framing = any(cue in text for cue in SUMMARY_FRAMING_CUES)
    reflective = any(cue in text for cue in SUMMARY_REFLECTIVE_CUES)
    return framing and reflective


def _has_shift_marker(text: str) -> bool:
    cues = [
        "ander onderwerp",
        "naast dat",
        "naast uw",
        "naast je",
        "ik wil ook nog vragen",
        "verder nog",
        "nu iets anders",
        "laten we ook kijken naar",
    ]
    return any(cue in text for cue in cues)


def _has_goal_alignment(text: str) -> bool:
    cues = [
        "het doel van dit gesprek",
        "vandaag wil ik",
        "ik wil graag",
        "we gaan bespreken",
        "waarvoor we hier zijn",
    ]
    return any(cue in text for cue in cues)


def _has_time_alignment(text: str) -> bool:
    cues = ["we hebben", "ongeveer", "minuten", "tijd", "duurt"]
    return any(cue in text for cue in cues)


def analyze_conversation_phases(
    turns: List[Dict[str, Any]], config: Dict[str, Any] = DEFAULT_PHASE_CONFIG
) -> Dict[str, Any]:
    """
    Main entry: detect rubric items per phase and return structured JSON.
    """
    phases = _init_phase_structure()
    metrics = _empty_metrics()
    evidence: List[Dict[str, Any]] = []
    if not turns:
        return {"phases": phases, "metrics": metrics, "evidence": evidence}

    normalized_turns: List[Dict[str, Any]] = []
    for turn in turns:
        text = str(turn.get("text", "") or "")
        norm = normalize_text(text)
        normalized_turns.append(
            {
                "speaker": (turn.get("speaker") or "").lower(),
                "text": text,
                "norm_text": norm,
                "tokens": content_words(norm, set(config["stopwords"])),
                "timestamp": turn.get("timestamp"),
            }
        )

    phase_map = segment_phases(normalized_turns, config)
    hit_map: Dict[str, Dict[str, Dict[str, int]]] = {}

    last_patient_tokens: Optional[List[str]] = None
    last_patient_idx: Optional[int] = None

    active_listening_recorded = 0

    for idx, turn in enumerate(normalized_turns):
        phase = phase_map.get(idx, "phase2")
        speaker = turn["speaker"]
        text = turn["norm_text"]
        raw_text = turn["text"]
        tokens = turn["tokens"]
        timestamp = turn.get("timestamp")

        if speaker == "patient":
            last_patient_tokens = tokens
            last_patient_idx = idx
            continue

        # Phase 1 detections
        if phase == "phase1":
            greeting = bool(
                re.search(r"\b(hallo|hoi|goedemorgen|goedemiddag|goedenavond)\b", text)
            )
            intro = bool(
                re.search(r"(ik ben|mijn naam is|ik heet|student verpleegkunde|ik loop stage)", text)
            )
            if greeting or intro:
                _record_hit(hit_map, phase, "greeting_intro", "weak")
                if greeting and intro:
                    _record_hit(hit_map, phase, "greeting_intro", "strong")
                _add_evidence(phases, phase, "greeting_intro", idx, raw_text, timestamp)
                _add_global_evidence(evidence, phase, "greeting_intro", idx, timestamp, raw_text, speaker)

            if _has_goal_alignment(text):
                strength = "strong" if _has_time_alignment(text) else "weak"
                _record_hit(hit_map, phase, "align_goal_content_time", strength)
                _add_evidence(phases, phase, "align_goal_content_time", idx, raw_text, timestamp)
                _add_global_evidence(evidence, phase, "align_goal_content_time", idx, timestamp, raw_text, speaker)

            if any(
                cue in text
                for cue in [
                    "hoe gaat het",
                    "hoe is het",
                    "fijn dat u er bent",
                    "zit u goed",
                    "heeft u het comfortabel",
                    "was de reis",
                    "goed geslapen",
                    "goed kunnen vinden",
                ]
            ):
                _record_hit(hit_map, phase, "small_talk_comfort", "weak")
                _add_evidence(phases, phase, "small_talk_comfort", idx, raw_text, timestamp)
                _add_global_evidence(evidence, phase, "small_talk_comfort", idx, timestamp, raw_text, speaker)

            if any(
                cue in text
                for cue in [
                    "wat verwacht u",
                    "waar hoopt u op",
                    "wat is belangrijk voor u",
                    "waar wilt u het over hebben",
                    "wat heeft u nodig",
                    "wat zijn uw verwachtingen",
                ]
            ):
                _record_hit(hit_map, phase, "needs_expectations", "strong")
                _add_evidence(phases, phase, "needs_expectations", idx, raw_text, timestamp)
                _add_global_evidence(evidence, phase, "needs_expectations", idx, timestamp, raw_text, speaker)

        # Phase 2 detections
        if phase == "phase2":
            is_open = _is_open_question(text)
            is_closed = _is_closed_question(text) and not is_open
            if is_open:
                metrics["open_q_count"] += 1
                _record_hit(hit_map, phase, "open_questions", "strong")
                _add_evidence(phases, phase, "open_questions", idx, raw_text, timestamp)
                _add_global_evidence(evidence, phase, "open_questions", idx, timestamp, raw_text, speaker)
            elif is_closed:
                metrics["closed_q_count"] += 1
                _record_hit(hit_map, phase, "closed_questions", "weak")
                _add_evidence(phases, phase, "closed_questions", idx, raw_text, timestamp)
                _add_global_evidence(evidence, phase, "closed_questions", idx, timestamp, raw_text, speaker)

            if _is_active_listening(text):
                if active_listening_recorded < config["active_listening_cap"]:
                    metrics["active_listening_count"] += 1
                    active_listening_recorded += 1
                    _record_hit(hit_map, phase, "active_listening", "weak")
                    _add_evidence(phases, phase, "active_listening", idx, raw_text, timestamp)
                    _add_global_evidence(evidence, phase, "active_listening", idx, timestamp, raw_text, speaker)

            if _has_understanding_check(text):
                metrics["understanding_check_count"] += 1
                _record_hit(hit_map, phase, "understanding_checks", "strong")
                _add_evidence(phases, phase, "understanding_checks", idx, raw_text, timestamp)
                _add_global_evidence(evidence, phase, "understanding_checks", idx, timestamp, raw_text, speaker)

            if _has_interim_summary(text):
                metrics["interim_summary_count"] += 1
                _record_hit(hit_map, phase, "interim_summaries", "strong")
                _add_evidence(phases, phase, "interim_summaries", idx, raw_text, timestamp)
                _add_global_evidence(evidence, phase, "interim_summaries", idx, timestamp, raw_text, speaker)

            if last_patient_tokens:
                overlap = jaccard_overlap(tokens, last_patient_tokens)
                continuation = any(
                    cue in text
                    for cue in [
                        "kunt u meer vertellen",
                        "hoe is dat",
                        "wat betekent dat",
                        "vertelt u verder",
                    ]
                )
                if overlap >= config["paraphrase_overlap"]:
                    strength = "strong" if continuation else "weak"
                    _record_hit(hit_map, phase, "paraphrase_repeat_to_continue", strength)
                    metrics["paraphrase_count"] += 1
                    _add_evidence(
                        phases, phase, "paraphrase_repeat_to_continue", idx, raw_text, timestamp
                    )
                    _add_global_evidence(
                        evidence,
                        phase,
                        "paraphrase_repeat_to_continue",
                        idx,
                        timestamp,
                        raw_text,
                        speaker,
                    )

                # E-in deepening: require explicit deepening cue
                if overlap >= config["e_in_overlap"] and any(
                    cue in text
                    for cue in [
                        "wat bedoelt u",
                        "kunt u uitleggen",
                        "kunt u een voorbeeld",
                        "hoe voelt dat",
                        "waardoor denkt u",
                        "wat maakt dat",
                    ]
                ):
                    metrics["e_in_count"] += 1
                    _record_hit(hit_map, phase, "e_in_deepen", "strong")
                    _add_evidence(phases, phase, "e_in_deepen", idx, raw_text, timestamp)
                    _add_global_evidence(
                        evidence, phase, "e_in_deepen", idx, timestamp, raw_text, speaker
                    )

                # E-ex topic shift
                if _has_shift_marker(text):
                    metrics["e_ex_count"] += 1
                    _record_hit(hit_map, phase, "e_ex_shift_topic", "strong")
                    _add_evidence(phases, phase, "e_ex_shift_topic", idx, raw_text, timestamp)
                    _add_global_evidence(
                        evidence, phase, "e_ex_shift_topic", idx, timestamp, raw_text, speaker
                    )
                elif overlap < config["e_ex_overlap_low"] and _is_open_question(text):
                    metrics["e_ex_count"] += 1
                    _record_hit(hit_map, phase, "e_ex_shift_topic", "weak")
                    _add_evidence(phases, phase, "e_ex_shift_topic", idx, raw_text, timestamp)
                    _add_global_evidence(
                        evidence, phase, "e_ex_shift_topic", idx, timestamp, raw_text, speaker
                    )

            # Content + emotional summary
            has_content = _has_content_summary(text)
            has_emotion = _has_emotional_reflection(text)
            if has_content or has_emotion:
                if has_content and has_emotion:
                    _record_hit(hit_map, phase, "content_and_emotional_summary", "strong")
                else:
                    _record_hit(hit_map, phase, "content_and_emotional_summary", "weak")
                if has_emotion:
                    metrics["emotional_reflection_count"] += 1
                _add_evidence(
                    phases, phase, "content_and_emotional_summary", idx, raw_text, timestamp
                )
                _add_global_evidence(
                    evidence,
                    phase,
                    "content_and_emotional_summary",
                    idx,
                    timestamp,
                    raw_text,
                    speaker,
                )

        # Phase 3 detections
        if phase == "phase3":
            if _has_content_summary(text):
                _record_hit(hit_map, phase, "end_summary", "strong")
                _add_evidence(phases, phase, "end_summary", idx, raw_text, timestamp)
                _add_global_evidence(evidence, phase, "end_summary", idx, timestamp, raw_text, speaker)

            if any(
                cue in text
                for cue in [
                    "we spreken af",
                    "de volgende stap",
                    "ik ga",
                    "u gaat",
                    "we plannen",
                    "ik verwijs",
                    "u krijgt",
                    "afspraak",
                    "controle",
                    "opvolging",
                ]
            ):
                _record_hit(hit_map, phase, "follow_up_agreements", "strong")
                _add_evidence(phases, phase, "follow_up_agreements", idx, raw_text, timestamp)
                _add_global_evidence(
                    evidence, phase, "follow_up_agreements", idx, timestamp, raw_text, speaker
                )

            if any(
                cue in text
                for cue in [
                    "heeft u nog vragen",
                    "bedankt",
                    "dank u wel",
                    "tot ziens",
                    "fijne dag",
                ]
            ):
                _record_hit(hit_map, phase, "professional_closing", "strong")
                _add_evidence(phases, phase, "professional_closing", idx, raw_text, timestamp)
                _add_global_evidence(
                    evidence, phase, "professional_closing", idx, timestamp, raw_text, speaker
                )

            if any(
                cue in text
                for cue in [
                    "wat vindt u",
                    "wat zou voor u werken",
                    "zullen we samen",
                    "hoe kijkt u daartegenaan",
                    "wat is voor u haalbaar",
                    "denkt u mee",
                ]
            ):
                _record_hit(hit_map, phase, "encourage_participation", "strong")
                _add_evidence(phases, phase, "encourage_participation", idx, raw_text, timestamp)
                _add_global_evidence(
                    evidence, phase, "encourage_participation", idx, timestamp, raw_text, speaker
                )

    _finalize_scores(phases, hit_map)
    return {"phases": phases, "metrics": metrics, "evidence": evidence}


# -----------------
# Gordon pattern detection
# -----------------


def detect_gordon_patterns(
    turns: List[Dict[str, Any]],
    config: Dict[str, Any] = DEFAULT_PHASE_CONFIG,
    pattern_config: Dict[str, Dict[str, Any]] = GORDON_PATTERNS,
) -> Dict[str, Any]:
    """
    Deterministic Gordon coverage detector based on keyword spotting.
    """
    result_patterns: Dict[str, Any] = {}
    total_patterns = len(pattern_config)
    stopwords = set(config["stopwords"])

    for pid, pdata in pattern_config.items():
        result_patterns[pid] = {
            "name": pdata["name"],
            "covered": False,
            "mention_count": 0,
            "source": None,
            "score": 0,
            "evidence": [],
        }

    def _kw_match(text: str, kw: str) -> bool:
        return re.search(rf"\b{re.escape(kw)}\b", text) is not None

    for idx, turn in enumerate(turns):
        text = normalize_text(str(turn.get("text", "") or ""))
        tokens = content_words(text, stopwords)
        speaker = (turn.get("speaker") or "").lower()
        timestamp = turn.get("timestamp")

        for pid, pdata in pattern_config.items():
            keywords = pdata["keywords"]
            matched_keywords = [kw for kw in keywords if _kw_match(text, kw)]
            if not matched_keywords:
                continue
            if speaker == "student" and not _looks_like_question(text):
                continue
            pattern_entry = result_patterns[pid]
            pattern_entry["mention_count"] += 1
            quote = turn.get("text", "").strip()
            negated = any(neg in text for neg in ["geen", "niet", "nooit"])
            evidence_entry = {
                "turn": idx,
                "timestamp": timestamp,
                "quote": quote,
                "speaker": speaker or "unknown",
            }
            if negated:
                evidence_entry["negated"] = True
            if len(pattern_entry["evidence"]) < 2:
                pattern_entry["evidence"].append(evidence_entry)
            if not pattern_entry["covered"]:
                pattern_entry["covered"] = True
                if pattern_entry["source"] and pattern_entry["source"] != speaker:
                    pattern_entry["source"] = "both"
                else:
                    pattern_entry["source"] = speaker if speaker else "unknown"
            else:
                if pattern_entry["source"] and speaker and pattern_entry["source"] != speaker:
                    pattern_entry["source"] = "both"

    covered_count = 0
    student_covered_count = 0
    for pid, entry in result_patterns.items():
        if entry["covered"]:
            covered_count += 1
            src = (entry.get("source") or "")
            is_student = (src == "student") or (src == "both") or ("student" in src)
            if is_student:
                student_covered_count += 1
                entry["score"] = 2
            else:
                entry["score"] = 1

    coverage_percent = (covered_count / total_patterns * 100) if total_patterns else 0.0
    student_coverage_percent = (
        student_covered_count / total_patterns * 100 if total_patterns else 0.0
    )

    return {
        "covered_count": covered_count,
        "total_patterns": total_patterns,
        "coverage_percent": round(coverage_percent, 1),
        "student_covered_count": student_covered_count,
        "student_coverage_percent": round(student_coverage_percent, 1),
        "patterns": result_patterns,
    }


# -----------------
# Renderer (optional teacher-style feedback)
# -----------------


WHY_MATTERS = {
    "greeting_intro": "Een duidelijke start zet de toon en geeft vertrouwen.",
    "align_goal_content_time": "Gezamenlijk doel en tijd afstemmen voorkomt misverstanden.",
    "small_talk_comfort": "Korte smalltalk verlaagt spanning en opent het gesprek.",
    "needs_expectations": "Door verwachtingen te peilen weet je waar de zorgvraag ligt.",
    "open_questions": "Open vragen geven de zorgvrager ruimte om te vertellen.",
    "closed_questions": "Gesloten vragen helpen feiten te verifiëren.",
    "active_listening": "Backchannels tonen dat je luistert zonder te onderbreken.",
    "understanding_checks": "Begripstoetsing voorkomt aannames.",
    "interim_summaries": "Tussentijds samenvatten geeft structuur en controle.",
    "paraphrase_repeat_to_continue": "Herhalen/parafraseren stimuleert de ander verder te praten.",
    "e_in_deepen": "E-in laat zien dat je dieper ingaat op wat de ander zei.",
    "e_ex_shift_topic": "E-ex markeert netjes een onderwerpwissel zonder abrupt te zijn.",
    "content_and_emotional_summary": "Combinatie van feiten en gevoel laat empathie en begrip zien.",
    "end_summary": "Een eindsamenvatting borgt gezamenlijke helderheid.",
    "follow_up_agreements": "Concrete afspraken maken het gesprek doelgericht.",
    "professional_closing": "Professioneel afsluiten laat een goede indruk achter.",
    "encourage_participation": "De zorgvrager betrekken vergroot eigen regie en therapietrouw.",
}

TRY_NEXT = {
    "greeting_intro": [
        "Begin met een warme begroeting en noem je naam en rol: \"Goedemiddag, ik ben Sam, student verpleegkunde.\""
    ],
    "align_goal_content_time": [
        "Leg kort het doel en de tijd uit: \"We hebben ongeveer 20 minuten om uw klachten te bespreken.\""
    ],
    "small_talk_comfort": [
        "Stel een korte comfortvraag: \"Zit u zo goed?\" of \"Hoe was de reis hierheen?\""
    ],
    "needs_expectations": [
        "Vraag expliciet naar verwachtingen: \"Wat hoopt u uit dit gesprek te halen?\""
    ],
    "open_questions": [
        "Gebruik open starters: \"Hoe merkt u dit in het dagelijks leven?\""
    ],
    "closed_questions": [
        "Gebruik gesloten vragen om feiten te verifiëren: \"Sinds wanneer heeft u deze pijn?\""
    ],
    "active_listening": [
        "Gebruik korte backchannels: \"Hmhm, ik luister\" of een korte \"ja\" zonder te onderbreken."
    ],
    "understanding_checks": [
        "Check begrip: \"Begrijp ik goed dat het vooral 's avonds speelt?\""
    ],
    "interim_summaries": [
        "Vat af en toe samen: \"Tot nu toe hoor ik dat...\""
    ],
    "paraphrase_repeat_to_continue": [
        "Herhaal kernwoorden en vraag door: \"U zegt dat het benauwd voelt; kunt u daar meer over vertellen?\""
    ],
    "e_in_deepen": [
        "Blijf bij hetzelfde onderwerp: \"Wat bedoelt u precies met duizelig?\""
    ],
    "e_ex_shift_topic": [
        "Markeer een nieuw onderwerp: \"Naast uw slaap, wil ik ook naar voeding vragen.\""
    ],
    "content_and_emotional_summary": [
        "Combineer inhoud en gevoel: \"U slaapt slecht en dat maakt u bezorgd; klopt dat?\""
    ],
    "end_summary": [
        "Sluit af met een korte samenvatting: \"Samenvattend: u ervaart...\""
    ],
    "follow_up_agreements": [
        "Maak de volgende stap concreet: \"We plannen een controle over twee weken.\""
    ],
    "professional_closing": [
        "Check vragen en bedank: \"Heeft u nog vragen? Dank u wel voor het gesprek.\""
    ],
    "encourage_participation": [
        "Vraag naar meedenken: \"Wat zou voor u haalbaar zijn als volgende stap?\""
    ],
}


def render_phase_feedback(analysis: Dict[str, Any]) -> str:
    """
    Render teacher-style feedback with checklist, evidence, and suggestions.
    """
    if not analysis:
        return ""
    phases = analysis.get("phases", {})
    lines: List[str] = []
    for phase_key in ["phase1", "phase2", "phase3"]:
        phase_data = phases.get(phase_key)
        if not phase_data:
            continue
        lines.append(PHASE_LABELS.get(phase_key, phase_key))
        items = phase_data.get("items", {})
        for item_key, item_data in items.items():
            score = item_data.get("score", 0)
            icon = "✓" if score == 2 else "△" if score == 1 else "✗"
            lines.append(f"- {icon} {ITEM_LABELS.get(item_key, item_key)}")
            ev_list = item_data.get("evidence") or []
            for ev in ev_list[:2]:
                quote = ev.get("quote", "")
                turn = ev.get("turn")
                lines.append(f"    • Turn {turn}: \"{quote}\"")
            why = WHY_MATTERS.get(item_key)
            if why:
                lines.append(f"    Waarom: {why}")
            suggestions = TRY_NEXT.get(item_key, [])
            if suggestions:
                lines.append(f"    Probeer: {suggestions[0]}")
        lines.append("")
    return "\n".join(lines).strip()
