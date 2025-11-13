"""
Format feedback into a clear, student-friendly structure.
"""


def format_student_feedback(conversation_feedback, gordon_result, speech_result):
    """
    Format all feedback components into a clear, student-friendly structure.
    
    Args:
        conversation_feedback: String feedback from Ollama about conversation quality
        gordon_result: Dict with Gordon pattern analysis results
        speech_result: Dict with speech analysis results (includes confidence)
        
    Returns:
        str: Formatted, student-friendly feedback text
    """
    
    sections = []
    
    # 1. QUICK SUMMARY (at the top - most important)
    summary_lines = []
    has_summary = False
    
    if speech_result and speech_result.get("confidence"):
        has_summary = True
        conf = speech_result["confidence"]
        score = conf.get("score", 0)
        level = conf.get("level", "medium")
        
        if level == "high":
            summary_lines.append(f"✅ **Zelfvertrouwen:** Hoog ({score}/100) - Je sprak met veel vertrouwen!")
        elif level == "medium":
            summary_lines.append(f"⚠️ **Zelfvertrouwen:** Gemiddeld ({score}/100) - Er is ruimte voor verbetering.")
        else:
            summary_lines.append(f"❌ **Zelfvertrouwen:** Laag ({score}/100) - Oefen met rustig en duidelijk spreken.")
    
    if gordon_result:
        has_summary = True
        covered = gordon_result.get("covered_patterns", 0)
        total = gordon_result.get("total_patterns", 11)
        percentage = gordon_result.get("coverage_percentage", 0)
        
        if percentage >= 60:
            summary_lines.append(f"✅ **Gezondheidspatronen:** {covered}/{total} besproken ({percentage}%) - Uitstekend!")
        elif percentage >= 30:
            summary_lines.append(f"⚠️ **Gezondheidspatronen:** {covered}/{total} besproken ({percentage}%) - Probeer meer aspecten te verkennen.")
        else:
            summary_lines.append(f"❌ **Gezondheidspatronen:** {covered}/{total} besproken ({percentage}%) - Meer variatie nodig.")
    
    if speech_result and speech_result.get("metrics"):
        has_summary = True
        metrics = speech_result["metrics"]
        filler_ratio = metrics.get("filler_ratio", 0)
        if filler_ratio < 3:
            summary_lines.append(f"✅ **Stopwoorden:** Weinig gebruikt - Goed gedaan!")
        elif filler_ratio < 10:
            summary_lines.append(f"⚠️ **Stopwoorden:** Enkele gebruikt - Probeer minder 'euh' of 'uh' te zeggen.")
        else:
            summary_lines.append(f"❌ **Stopwoorden:** Veel gebruikt ({filler_ratio}%) - Oefen met rustig spreken.")
    
    if has_summary:
        sections.append("## 📊 Samenvatting van je prestaties\n" + "\n".join(summary_lines))
    
    # 2. CONVERSATION FEEDBACK (main content feedback from Ollama)
    if conversation_feedback:
        sections.append("\n## 💬 Gespreksvaardigheden\n" + conversation_feedback)
    
    # 3. CONFIDENCE & SPEAKING TIPS (if available)
    if speech_result and speech_result.get("confidence"):
        speech_section = ["## 🎤 Spreekvaardigheden\n"]
        
        conf = speech_result["confidence"]
        # Confidence explanation
        score = conf.get("score", 0)
        level = conf.get("level", "medium")
        indicators = conf.get("indicators", [])
        
        confidence_explanation = ""
        if level == "high":
            confidence_explanation = f"**Je zelfvertrouwen:** {score}/100 - Je sprak met veel zelfvertrouwen! Je spreekpatroon was natuurlijk en vloeiend.\n"
        elif level == "medium":
            confidence_explanation = f"**Je zelfvertrouwen:** {score}/100 - Je sprak met redelijk zelfvertrouwen. Er is ruimte voor verbetering.\n"
        else:
            confidence_explanation = f"**Je zelfvertrouwen:** {score}/100 - Je sprak met weinig zelfvertrouwen. Oefen met rustig en duidelijk spreken.\n"
        
        speech_section.append(confidence_explanation)
        
        # Indicators (what contributed to the score)
        if indicators:
            speech_section.append("**Waarom dit score:**\n")
            for indicator in indicators[:5]:  # Show top 5
                speech_section.append(f"- {indicator}\n")
        
        # Speaking tips
        if speech_result.get("summary"):
            speaking_tips = speech_result["summary"].strip()
            # Remove the confidence explanation if it's already in the summary
            if "Je sprak met" in speaking_tips:
                # Split and take everything after the first sentence
                tip_parts = speaking_tips.split(". ", 1)
                if len(tip_parts) > 1:
                    speaking_tips = tip_parts[1]
            
            if speaking_tips and speaking_tips != confidence_explanation.strip():
                speech_section.append("\n**Tips om te verbeteren:**\n")
                speech_section.append(speaking_tips)
        
        sections.append("\n".join(speech_section))
    
    # 4. GORDON PATTERNS (what health topics were covered)
    if gordon_result and gordon_result.get("summary"):
        sections.append("\n## 🏥 Gezondheidspatronen Analyse\n")
        sections.append(gordon_result["summary"])
    
    # 5. ACTION ITEMS (clear next steps)
    action_items = []
    
    if speech_result and speech_result.get("confidence"):
        conf_level = speech_result["confidence"].get("level", "medium")
        if conf_level == "low":
            action_items.append("🔹 Oefen met rustig en duidelijk spreken om je zelfvertrouwen te vergroten")
        
        metrics = speech_result.get("metrics", {})
        if metrics.get("filler_ratio", 0) > 5:
            action_items.append("🔹 Probeer minder stopwoorden te gebruiken (zoals 'euh', 'uh')")
        
        if metrics.get("speech_rate_wpm", 0) < 80:
            action_items.append("🔹 Probeer iets sneller te spreken voor een natuurlijker tempo")
        elif metrics.get("speech_rate_wpm", 0) > 150:
            action_items.append("🔹 Probeer iets langzamer te spreken voor meer duidelijkheid")
    
    if gordon_result:
        coverage = gordon_result.get("coverage_percentage", 0)
        if coverage < 50:
            action_items.append("🔹 Probeer meer verschillende gezondheidsaspecten te bespreken (zoals voeding, slaap, beweging)")
        
        missing = gordon_result.get("mentioned_patterns", [])
        if len(missing) < 6:
            action_items.append("🔹 Stel vragen over verschillende gezondheidsaspecten (pijn, familie, stress, voeding, etc.)")
    
    if action_items:
        sections.append("\n## 📋 Actiepunten voor volgende keer\n")
        sections.append("\n".join(action_items))
    
    # Combine all sections
    formatted_feedback = "\n\n".join(sections)
    
    return formatted_feedback

