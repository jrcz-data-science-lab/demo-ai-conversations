import random
import statistics

# Filler words and hesitation markers for both Dutch and English
# Rule 2: Specific hesitation markers that MUST be detected
HESITATION_MARKERS = ["eh", "eeh", "ehm", "uh", "uhm", "mmm", "hmm", "emmm", "euh", "ah", "um"]
DUTCH_FILLERS = ["euh", "eh", "uh", "ah", "um", "nou", "dus", "hmm", "ehm"]
ENGLISH_FILLERS = ["uh", "um", "ah", "er", "so", "like", "you know", "well", "hmm"]
ALL_FILLERS = list(set(HESITATION_MARKERS + DUTCH_FILLERS + ENGLISH_FILLERS))

# Positive compliments for good performance
POSITIVE_COMPLIMENTS = [
    "Mooi rustige toon, prettig om naar te luisteren.",
    "Je tempo klonk verzorgd en warm.",
    "Goed gedaan met het behouden van een gelijkmatige flow.",
    "Je sprak helder en natuurlijk.",
    "Sterke controle over je spreektempo.",
    "Je articulatie was duidelijk en betrokken."
]


def analyze_speech_patterns(audio_metadata_list):
    """
    Analyze speech patterns from audio metadata.
    
    Args:
        audio_metadata_list: List of dicts containing audio metadata for the session
        
    Returns:
        dict: Metrics including filler_count, filler_ratio, avg_pause, speech_rate_wpm, etc.
    """
    if not audio_metadata_list:
        # Return default metrics if no data
        return {
            "speech_rate_wpm": 0,
            "avg_pause": 0,
            "filler_count": 0,
            "filler_ratio": 0,
            "long_pause_count": 0,
            "total_words": 0,
            "total_duration": 0
        }
    
    total_words = 0
    total_duration = 0
    all_segments = []
    all_transcript_texts = []  # Collect all transcript texts for filler counting
    
    # Collect all data from metadata
    for metadata in audio_metadata_list:
        word_count = metadata.get("word_count", 0)
        audio_duration = metadata.get("audio_duration", 0)
        transcript_details = metadata.get("transcript_details", {})
        
        total_words += word_count
        total_duration += audio_duration or 0
        
        # Extract segments for pause analysis
        segments = transcript_details.get("segments", [])
        if segments:
            all_segments.extend(segments)
            # Extract transcript text from segments for filler counting
            transcript_text = " ".join([seg.get("text", "") for seg in segments])
            if transcript_text:
                all_transcript_texts.append(transcript_text)
    
    # Count filler words across all transcripts (Rule 2: detect hesitation markers)
    filler_count = 0
    filler_details = []  # Store exact filler occurrences for quoting
    combined_text = " ".join(all_transcript_texts).lower()
    if combined_text:
        words = combined_text.split()
        for idx, word in enumerate(words):
            # Remove punctuation for matching
            clean_word = word.strip(".,!?;:()[]{}'\"")
            # Check for exact hesitation markers (priority) or other fillers
            if clean_word in HESITATION_MARKERS:
                filler_count += 1
                # Store context (previous and next word) for quoting
                prev_word = words[idx - 1] if idx > 0 else ""
                next_word = words[idx + 1] if idx < len(words) - 1 else ""
                filler_details.append({
                    "filler": clean_word,
                    "context": f"{prev_word} {clean_word} {next_word}".strip(),
                    "index": idx
                })
            elif clean_word in ALL_FILLERS:
                filler_count += 1
    
    # Calculate speaking rate (words per minute)
    if total_duration > 0:
        speech_rate_wpm = (total_words / total_duration) * 60
    else:
        speech_rate_wpm = 0
    
    # Calculate pause analysis from segments
    pause_durations = []
    long_pause_count = 0
    
    if len(all_segments) > 1:
        # Sort segments by start time
        sorted_segments = sorted(all_segments, key=lambda x: x.get("start", 0))
        
        for i in range(len(sorted_segments) - 1):
            current_end = sorted_segments[i].get("end", 0)
            next_start = sorted_segments[i + 1].get("start", 0)
            
            if next_start > current_end:
                pause_duration = next_start - current_end
                pause_durations.append(pause_duration)
                
                if pause_duration > 2.0:  # Long pause threshold
                    long_pause_count += 1
    
    # Calculate average pause
    if pause_durations:
        avg_pause = statistics.mean(pause_durations)
    else:
        # Fallback: estimate from word spacing if no timestamps
        if total_words > 0 and total_duration > 0:
            # Estimate average pause as small gaps (rough approximation)
            avg_pause = (total_duration / total_words) * 0.1  # Assume 10% of time is pauses
        else:
            avg_pause = 0
    
    # Calculate filler ratio (fillers per 100 words)
    filler_ratio = (filler_count / total_words * 100) if total_words > 0 else 0
    
    # Calculate filler density per 10 words
    filler_density_per_10_words = (filler_count / total_words * 10) if total_words > 0 else 0
    
    return {
        "speech_rate_wpm": round(speech_rate_wpm, 1),
        "avg_pause": round(avg_pause, 2),
        "filler_count": filler_count,
        "filler_ratio": round(filler_ratio, 2),
        "filler_density_per_10_words": round(filler_density_per_10_words, 2),
        "filler_details": filler_details[:10],  # Keep first 10 for reporting
        "hesitation_markers": filler_count,  # Use same count for backward compatibility
        "long_pause_count": long_pause_count,
        "total_words": total_words,
        "total_duration": round(total_duration, 2)
    }


def interpret_metrics(metrics):
    """
    Convert numeric metrics to human-readable feedback sentences.
    
    Args:
        metrics: dict with speech pattern metrics
        
    Returns:
        list: List of feedback sentences
    """
    feedback = []
    
    filler_ratio = metrics.get("filler_ratio", 0)
    speech_rate_wpm = metrics.get("speech_rate_wpm", 0)
    avg_pause = metrics.get("avg_pause", 0)
    long_pause_count = metrics.get("long_pause_count", 0)
    
    # Filler word feedback
    if filler_ratio > 10:
        feedback.append("Je hebt veel stopwoorden gebruikt zoals 'euh' of 'eh'. Door ze te schrappen klinkt je verhaal helderder.")
    elif filler_ratio > 5:
        feedback.append("Je gebruikte enkele stopwoorden. Neem kort een ademhaling in plaats van een 'euh'.")
    elif filler_ratio > 0:
        feedback.append("Je gebruikte weinig stopwoorden, goed gedaan.")
    
    # Speaking rate feedback
    if speech_rate_wpm > 150:
        feedback.append("Je sprak vrij snel. Iets rustiger helpt de patiënt om mee te komen.")
    elif speech_rate_wpm < 80 and speech_rate_wpm > 0:
        feedback.append("Je sprak vrij langzaam. Verhoog het tempo iets voor een natuurlijker gesprek.")
    elif 100 <= speech_rate_wpm <= 140:
        feedback.append("Je spreektempo was goed en natuurlijk.")
    
    # Pause feedback
    if avg_pause < 0.3:
        feedback.append("Voeg korte pauzes toe tussen ideeën om natuurlijker te klinken.")
    elif avg_pause > 2.0:
        feedback.append("Je had enkele lange pauzes. Probeer een constantere flow te behouden.")
    else:
        if avg_pause > 0:
            feedback.append("Je pauzes tussen zinnen waren goed gebalanceerd.")
    
    # Long pause feedback
    if long_pause_count > 3:
        feedback.append("Je had meerdere lange pauzes. Probeer een vloeiender spraakritme te behouden.")
    
    # If no specific feedback, add positive general feedback
    if not feedback:
        feedback.append("Je spreekpatroon was goed. Blijf zo doorgaan!")
    
    return feedback


def add_positive_compliments(metrics, existing_feedback):
    """
    Add positive compliments when metrics are mostly good.
    
    Args:
        metrics: dict with speech pattern metrics
        existing_feedback: list of existing feedback sentences
        
    Returns:
        list: Feedback list with compliments added
    """
    filler_ratio = metrics.get("filler_ratio", 0)
    speech_rate_wpm = metrics.get("speech_rate_wpm", 0)
    avg_pause = metrics.get("avg_pause", 0)
    
    # Check if metrics are mostly good
    is_good_filler = filler_ratio < 8
    is_good_pace = 100 <= speech_rate_wpm <= 140
    is_good_pause = 0.3 <= avg_pause <= 2.0
    
    # Add compliment if at least 2 out of 3 metrics are good
    good_count = sum([is_good_filler, is_good_pace, is_good_pause])
    if good_count >= 2:
        compliment = random.choice(POSITIVE_COMPLIMENTS)
        # Add at the beginning for visibility
        existing_feedback.insert(0, compliment)
    
    return existing_feedback


OPEN_QUESTION_PREFIXES = ["wat", "hoe", "waar", "waarom", "kunt u", "kunt je", "kan u", "kan je"]
EMPATHY_CUES = ["vervelend", "kan me voorstellen", "klinkt lastig", "spijtig", "snap dat"]
GORDON_TOPIC_HINTS = {
    "Slaap": ["slaap", "slapen", "nacht", "vermoeid"],
    "Stress": ["stress", "spanning", "coping", "zorgen"],
    "Voeding": ["eten", "drink", "voeding", "eetlust"],
    "Activiteit": ["bewegen", "lopen", "trap", "sport", "wandelen"],
    "Uitscheiding": ["plassen", "ontlasting", "toilet"],
    "Gezondheidsbeleving": ["gezondheid", "klachten", "pijn"],
    "Cognitie": ["denken", "geheugen", "concentratie"],
}


def extract_conversation_cues(conversation_history):
    """
    Lightweight scan of the transcript to ground feedback in actual statements.
    """
    cues = {
        "first_student_line": None,
        "open_questions": [],
        "closed_questions": [],
        "empathy_examples": [],
        "gordon_mentions": [],
        "student_word_count": 0,
    }
    if not conversation_history:
        return cues

    student_messages = []
    for line in conversation_history.splitlines():
        if line.strip().startswith("Student:"):
            msg = line.split("Student:", 1)[1].strip()
            if msg:
                student_messages.append(msg)

    if not student_messages:
        return cues

    cues["first_student_line"] = student_messages[0]
    cues["student_word_count"] = sum(len(msg.split()) for msg in student_messages)

    gordon_hits = set()
    for msg in student_messages:
        lower = msg.lower()
        cleaned = lower.strip(" .,!?:;")
        if any(cleaned.startswith(prefix + " ") or cleaned == prefix for prefix in OPEN_QUESTION_PREFIXES):
            cues["open_questions"].append(msg)
        elif cleaned.endswith("?"):
            cues["closed_questions"].append(msg)

        if any(token in lower for token in EMPATHY_CUES):
            cues["empathy_examples"].append(msg)

        for pattern, keywords in GORDON_TOPIC_HINTS.items():
            if any(keyword in lower for keyword in keywords):
                gordon_hits.add(pattern)

    cues["gordon_mentions"] = sorted(gordon_hits)
    return cues


def build_speaking_summary(metrics, cues):
    """
    Create the nursing-teacher style Speaking Tips Summary.
    """
    speech_rate = metrics.get("speech_rate_wpm", 0)
    avg_pause = metrics.get("avg_pause", 0)
    filler_count = metrics.get("filler_count", 0)

    first_line = cues.get("first_student_line") or ""
    opener = first_line[:160] + ("..." if len(first_line) > 160 else "")

    open_questions = cues.get("open_questions") or []
    closed_questions = cues.get("closed_questions") or []
    empathy_examples = cues.get("empathy_examples") or []
    gordon_mentions = cues.get("gordon_mentions") or []

    # Bullet 1: houding/gespreksstijl
    if opener:
        bullet1 = f"Je opende met \"{opener}\", dat geeft direct contact; houd die warme toon vast zoals in een anamnesegesprek."
    elif empathy_examples:
        bullet1 = f"Je toonde betrokkenheid met zinnen als \"{empathy_examples[0]}\", dat helpt om veilig contact te maken."
    else:
        bullet1 = "Je toon was verzorgd; blijf expliciet contact maken zoals een docent in de anamnese zou coachen."

    # Bullet 2: tempo/pauzes
    if speech_rate <= 0:
        bullet2 = "Het spreektempo was niet meetbaar; kies een rustig tempo en laat na elke vraag kort stilte vallen."
    else:
        tempo_note = f"Je tempo lag rond {speech_rate:.0f} wpm met pauzes van {avg_pause:.2f}s."
        if speech_rate > 170 or avg_pause < 0.35:
            bullet2 = f"{tempo_note} In een anamnesegesprek is het belangrijk om wat meer rust te laten zodat de patiënt kan reageren."
        elif avg_pause > 2.2:
            bullet2 = f"{tempo_note} Laat de pauzes korter en gericht zijn, zodat de flow van het gesprek blijft."
        else:
            bullet2 = f"{tempo_note} Dat geeft meestal voldoende rust; blijf na elke vraag kort wachten."

    # Bullet 3: klinische communicatie
    if open_questions:
        bullet3 = f"Je gebruikte open vragen zoals \"{open_questions[0]}\"; mooi voor actief luisteren en verkennen van patronen."
    elif closed_questions:
        bullet3 = f"Je stelde vooral gesloten vragen zoals \"{closed_questions[0]}\"; voeg meer open vragen toe voor verdiepend anamnesewerk."
    else:
        bullet3 = "Ik zag weinig verkennende vragen; stel open vragen om de patiënt zelf te laten vertellen en emoties te delen."

    # Bullet 4: gesprekstechniek + advies
    if filler_count > 0:
        bullet4 = f"Je noteerde {filler_count} fillers; vervang die door een korte stilte en vraag daarna door op thema's als {', '.join(gordon_mentions[:2]) or 'slaap of stress'}."
    elif gordon_mentions:
        bullet4 = f"Je raakte {', '.join(gordon_mentions[:2])} al aan; stel na elke vraag een verdiepende vraag en laat even stilte vallen."
    else:
        bullet4 = "Probeer na elke vraag twee tellen stilte te laten en koppel terug naar een volgend Gordon-patroon, bijvoorbeeld slaap of coping."

    lines = [
        "=== Speaking Tips Summary ===",
        f"• {bullet1}",
        f"• {bullet2}",
        f"• {bullet3}",
        f"• {bullet4}",
    ]
    return "\n".join(lines)




def generate_icon_states(metrics):
    """
    Generate icon state labels for Unreal Engine visualization.
    
    Args:
        metrics: dict with speech pattern metrics
        
    Returns:
        dict: Icon states for Unreal Engine
    """
    speech_rate_wpm = metrics.get("speech_rate_wpm", 0)
    avg_pause = metrics.get("avg_pause", 0)
    filler_ratio = metrics.get("filler_ratio", 0)
    
    # Determine speech rate state
    if speech_rate_wpm > 150:
        speech_rate_state = "fast"
    elif speech_rate_wpm < 80:
        speech_rate_state = "slow"
    else:
        speech_rate_state = "normal"
    
    # Determine pause state
    if avg_pause < 0.3:
        pause_state = "short"
    elif avg_pause > 2.0:
        pause_state = "long"
    else:
        pause_state = "balanced"
    
    # Determine filler state
    if filler_ratio > 10:
        filler_state = "many"
    else:
        filler_state = "few"
    
    # Determine overall state
    if filler_ratio < 8 and 100 <= speech_rate_wpm <= 140:
        overall_state = "good"
    else:
        overall_state = "needs_improvement"
    
    return {
        "speech_rate": speech_rate_state,
        "pauses": pause_state,
        "fillers": filler_state,
        "overall": overall_state
    }


def generate_speech_feedback(audio_metadata_list, conversation_history=None):
    """
    Main function to generate complete speech pattern feedback.
    
    Args:
        audio_metadata_list: List of dicts containing audio metadata for the session
        conversation_history: Optional conversation text for content quality analysis
        
    Returns:
        dict: Complete feedback including metrics, interpretations, summary, and icon_states
    """
    # Analyze speech patterns
    metrics = analyze_speech_patterns(audio_metadata_list)

    # Interpret metrics into human feedback
    interpretations = interpret_metrics(metrics)
    interpretations = add_positive_compliments(metrics, interpretations)

    conversation_cues = extract_conversation_cues(conversation_history)
    summary = build_speaking_summary(metrics, conversation_cues)

    # Generate icon states for Unreal Engine
    icon_states = generate_icon_states(metrics)

    return {
        "metrics": metrics,
        "interpretations": interpretations,
        "summary": summary,
        "icon_states": icon_states,
        "conversation_cues": conversation_cues,
    }
