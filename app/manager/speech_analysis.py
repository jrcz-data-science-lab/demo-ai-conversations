import json
import random
import statistics

# Filler words for both Dutch and English
DUTCH_FILLERS = ["euh", "eh", "uh", "ah", "um", "nou", "dus", "hmm", "ehm"]
ENGLISH_FILLERS = ["uh", "um", "ah", "er", "so", "like", "you know", "well", "hmm"]
ALL_FILLERS = DUTCH_FILLERS + ENGLISH_FILLERS

# Positive compliments for good performance
POSITIVE_COMPLIMENTS = [
    "Nice pacing overall!",
    "Your tone sounded calm and confident.",
    "Good job maintaining a steady flow.",
    "You spoke clearly and naturally.",
    "Excellent control of your speaking pace.",
    "You sounded confident and articulate."
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
    
    # Count filler words across all transcripts
    filler_count = 0
    combined_text = " ".join(all_transcript_texts).lower()
    if combined_text:
        words = combined_text.split()
        for word in words:
            # Remove punctuation for matching
            clean_word = word.strip(".,!?;:()[]{}'\"")
            if clean_word in ALL_FILLERS:
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
    
    return {
        "speech_rate_wpm": round(speech_rate_wpm, 1),
        "avg_pause": round(avg_pause, 2),
        "filler_count": filler_count,
        "filler_ratio": round(filler_ratio, 2),
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
        feedback.append("Je hebt veel stopwoorden gebruikt zoals 'euh' of 'eh'. Probeer deze te verminderen voor meer zelfvertrouwen.")
    elif filler_ratio > 5:
        feedback.append("Je gebruikte enkele stopwoorden. Probeer rustiger te spreken en kort te pauzeren in plaats van 'euh' te zeggen.")
    elif filler_ratio > 0:
        feedback.append("Je gebruikte weinig stopwoorden, goed gedaan!")
    
    # Speaking rate feedback
    if speech_rate_wpm > 150:
        feedback.append("Je sprak vrij snel. Door wat langzamer te spreken klink je duidelijker en zelfverzekerder.")
    elif speech_rate_wpm < 80 and speech_rate_wpm > 0:
        feedback.append("Je sprak vrij langzaam. Probeer het tempo iets te verhogen voor een natuurlijker gesprek.")
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


def calculate_confidence_score(metrics):
    """
    Calculate confidence score (0-100) based on speech patterns.
    
    Higher confidence indicators:
    - Normal speaking rate (100-140 WPM)
    - Low filler ratio (< 5%)
    - Balanced pauses (0.3-2.0s)
    - Few long pauses
    
    Lower confidence indicators:
    - Very slow or fast speaking rate
    - High filler ratio
    - Extremely short or long pauses
    - Many long pauses
    
    Args:
        metrics: dict with speech pattern metrics
        
    Returns:
        dict: {
            "score": int (0-100),
            "level": str ("high", "medium", "low"),
            "indicators": list of confidence indicators
        }
    """
    filler_ratio = metrics.get("filler_ratio", 0)
    speech_rate_wpm = metrics.get("speech_rate_wpm", 0)
    avg_pause = metrics.get("avg_pause", 0)
    long_pause_count = metrics.get("long_pause_count", 0)
    total_duration = metrics.get("total_duration", 0)
    
    # Calculate long pause ratio (long pauses per minute)
    if total_duration > 0:
        long_pause_ratio = (long_pause_count / total_duration) * 60
    else:
        long_pause_ratio = 0
    
    score = 100  # Start with perfect score
    indicators = []
    
    # Speaking rate scoring (max -30 points)
    if 100 <= speech_rate_wpm <= 140:
        # Perfect range: no penalty
        indicators.append("Goed spreektempo")
    elif 80 <= speech_rate_wpm < 100 or 140 < speech_rate_wpm <= 160:
        # Slightly off: -10 points
        score -= 10
        indicators.append("Spreektempo iets afwijkend")
    elif speech_rate_wpm < 80:
        # Too slow (can indicate nervousness): -20 points
        score -= 20
        indicators.append("Zeer langzaam spreektempo (mogelijk onzekerheid)")
    elif speech_rate_wpm > 160:
        # Too fast (can indicate nervousness): -20 points
        score -= 20
        indicators.append("Zeer snel spreektempo (mogelijk nervositeit)")
    elif speech_rate_wpm == 0:
        # No data
        score -= 15
    
    # Filler ratio scoring (max -40 points)
    if filler_ratio < 3:
        # Excellent: no penalty
        indicators.append("Weinig stopwoorden")
    elif 3 <= filler_ratio < 5:
        # Good: -5 points
        score -= 5
        indicators.append("Enkele stopwoorden")
    elif 5 <= filler_ratio < 10:
        # Moderate: -15 points
        score -= 15
        indicators.append("Veel stopwoorden (wijst op onzekerheid)")
    elif filler_ratio >= 10:
        # High: -30 points
        score -= 30
        indicators.append("Zeer veel stopwoorden (wijst op nervositeit)")
    
    # Pause scoring (max -20 points)
    if 0.3 <= avg_pause <= 2.0:
        # Balanced pauses: no penalty
        indicators.append("Goede pauzes")
    elif avg_pause < 0.3:
        # Too short (rushed): -10 points
        score -= 10
        indicators.append("Te korte pauzes (mogelijk haast/onzekerheid)")
    elif avg_pause > 2.0:
        # Too long (thinking, uncertainty): -15 points
        score -= 15
        indicators.append("Lange pauzes (mogelijk onzekerheid)")
    
    # Long pause scoring (max -10 points)
    if long_pause_ratio < 1:
        # Few long pauses: no penalty
        pass
    elif 1 <= long_pause_ratio < 3:
        # Some long pauses: -5 points
        score -= 5
        indicators.append("Enkele lange pauzes")
    elif long_pause_ratio >= 3:
        # Many long pauses: -10 points
        score -= 10
        indicators.append("Veel lange pauzes (wijst op onzekerheid)")
    
    # Ensure score is between 0-100
    score = max(0, min(100, score))
    
    # Determine confidence level
    if score >= 75:
        level = "high"
    elif score >= 50:
        level = "medium"
    else:
        level = "low"
    
    return {
        "score": round(score),
        "level": level,
        "indicators": indicators
    }


def interpret_confidence(confidence_result):
    """
    Generate human-readable feedback based on confidence score.
    
    Args:
        confidence_result: dict from calculate_confidence_score()
        
    Returns:
        str: Feedback message about confidence level
    """
    score = confidence_result.get("score", 50)
    level = confidence_result.get("level", "medium")
    
    if level == "high":
        return f"Je sprak met veel zelfvertrouwen (score: {score}/100). Je spreekpatroon was natuurlijk en vloeiend."
    elif level == "medium":
        return f"Je sprak met redelijk zelfvertrouwen (score: {score}/100). Er is ruimte voor verbetering in je spreekpatroon."
    else:
        return f"Je sprak met weinig zelfvertrouwen (score: {score}/100). Oefen met rustig en duidelijk spreken om je zelfvertrouwen te vergroten."


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


def generate_speech_feedback(audio_metadata_list):
    """
    Main function to generate complete speech pattern feedback.
    
    Args:
        audio_metadata_list: List of dicts containing audio metadata for the session
        
    Returns:
        dict: Complete feedback including metrics, interpretations, summary, and icon_states
    """
    # Analyze speech patterns
    metrics = analyze_speech_patterns(audio_metadata_list)
    
    # Calculate confidence score
    confidence_result = calculate_confidence_score(metrics)
    
    # Interpret metrics into human feedback
    interpretations = interpret_metrics(metrics)
    
    # Add confidence feedback at the beginning
    confidence_feedback = interpret_confidence(confidence_result)
    interpretations.insert(0, confidence_feedback)
    
    # Add positive compliments if metrics are good
    interpretations = add_positive_compliments(metrics, interpretations)
    
    # Generate summary
    if interpretations:
        summary = " ".join(interpretations)
    else:
        summary = "Je spreekpatroon was goed. Blijf zo doorgaan!"
    
    # Generate icon states for Unreal Engine
    icon_states = generate_icon_states(metrics)
    
    # Add confidence to icon states
    icon_states["confidence"] = confidence_result.get("level", "medium")
    
    return {
        "metrics": metrics,
        "confidence": confidence_result,
        "interpretations": interpretations,
        "summary": summary,
        "icon_states": icon_states
    }

