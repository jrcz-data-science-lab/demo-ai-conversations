from fpdf import FPDF
from pathlib import Path


def extract_student_view(data: dict) -> dict:
    meta = data["structured"]["metadata"]
    gordon = data["structured"]["gordon_detected"]
    phases = data["structured"]["phase_analysis"]["phases"]
    actions = data["structured"]["sections"]["action_items"]
    summary = data["structured"]["sections"]["summary"]
    closing = data["structured"]["sections"]["closing"]

    # Extract strengths / improvements / concrete question safely
    strengths = actions.split("**Sterke punten")[1].split("**Verbeterpunten")[0].strip()
    improvements = actions.split("**Verbeterpunten")[1].split("**Specifieke")[0].strip()
    next_question = actions.split("**Concrete vraag om te stellen:**")[1].strip()

    return {
        "header": {
            "speech_rate": meta["speech_rate_wpm"],
            "prosody": meta["prosody_score"],
            "emotion": meta["emotion"],
            "coverage": gordon["coverage_percent"]
        },
        "summary": summary,
        "gordon_patterns": gordon["patterns"],
        "phases": phases,
        "strengths": strengths,
        "improvements": improvements,
        "next_question": next_question,
        "closing": closing
    }


def create_feedback_pdf(data: dict, filepath: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = Path("/usr/src/app/email_utils/dejavu-sans/DejaVuSans.ttf")
    if not font_path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")

    pdf.add_font("dejavu", "", str(font_path))
    pdf.set_font("dejavu", size=11)

    student = extract_student_view(data)

    def add_section(title, text):
        pdf.set_font("dejavu", size=12)
        pdf.multi_cell(0, 8, title)
        pdf.ln(1)
        pdf.set_font("dejavu", size=11)
        pdf.multi_cell(0, 7, text)
        pdf.ln(4)

    # ---------- Header ----------
    h = student["header"]
    add_section("Gespreksprofiel",
        f"Spreeksnelheid: {h['speech_rate']} wpm\n"
        f"Prosodie: {h['prosody']}/100\n"
        f"Emotionele toon: {h['emotion']}\n"
        f"Gordon-dekking: {h['coverage']}%"
    )

    # ---------- Summary ----------
    add_section("Samenvatting", student["summary"])

    # ---------- Gordon overview ----------
    pdf.set_font("dejavu", size=12)
    pdf.multi_cell(0, 8, "Gordon-patronen overzicht")
    pdf.set_font("dejavu", size=11)

    for p in student["gordon_patterns"].values():
        status = "✓" if p["covered"] else "✗"
        pdf.multi_cell(0, 6, f"{status} {p['name']}")

    pdf.ln(4)

    # ---------- Core skills ----------
    pdf.set_font("dejavu", size=12)
    pdf.multi_cell(0, 8, "Gespreksvaardigheden (kern)")
    pdf.set_font("dejavu", size=11)

    phases = student["phases"]
    core_skills = {
        "Begroeting": phases["phase1"]["items"]["greeting_intro"]["present"],
        "Open vragen": phases["phase2"]["items"]["open_questions"]["present"],
        "Begrip checken": phases["phase2"]["items"]["understanding_checks"]["present"],
        "Parafraseren": phases["phase2"]["items"]["paraphrase_repeat_to_continue"]["present"],
        "Eindsamenvatting": phases["phase3"]["items"]["end_summary"]["present"],
        "Professioneel afsluiten": phases["phase3"]["items"]["professional_closing"]["present"],
    }

    for skill, ok in core_skills.items():
        pdf.multi_cell(0, 6, f"{'✓' if ok else '✗'} {skill}")

    pdf.ln(4)

    # ---------- Strengths & improvements ----------
    add_section("Sterke punten", student["strengths"])
    add_section("Verbeterpunten", student["improvements"])

    # ---------- Next action ----------
    add_section("Concrete vraag voor volgende keer", student["next_question"])

    # ---------- Closing ----------
    add_section("Afsluiting", student["closing"])

    pdf.output(filepath)
