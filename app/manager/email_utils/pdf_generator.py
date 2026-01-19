from fpdf import FPDF
from pathlib import Path

def create_feedback_pdf(data: dict, filepath: str, logo_path: str | None = None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    regular_font_path = Path("/usr/src/app/email_utils/dejavu-sans/DejaVuSans.ttf")
    bold_font_path = Path("/usr/src/app/email_utils/dejavu-sans/DejaVuSans-Bold.ttf")

    # Add font
    pdf.add_font("dejavu-sans", style="", fname=str(regular_font_path))
    pdf.add_font("dejavu-sans", style='B', fname=str(bold_font_path))
    
    # Set the font for the document content (regular style)
    pdf.set_font("dejavu-sans", style="", size=11)

    # Add some title or header text if necessary
    pdf.set_font("dejavu-sans", 'B', size=16)
    pdf.cell(200, 10, txt="Feedback Document", ln=True, align="C")
    pdf.ln(10)

    # Add the content sections
    s = data["structured"]["sections"]

    def add_section(title, text):
        pdf.set_font("dejavu-sans", style='B', size=14)
        pdf.multi_cell(0, 10, title)
        pdf.ln(2)
        
        # Set content font (regular style, normal size)
        pdf.set_font("dejavu-sans", style="", size=11)
        pdf.multi_cell(0, 8, text)
        pdf.ln(6)

    # Add all sections with improved styling
    add_section("1. Samenvatting", s["summary"])
    add_section("2. Gespreksvaardigheden", s["gespreksvaardigheden"])
    add_section("3. Begripstoetsing", s["comprehension"])
    add_section("4. Fase feedback", s["phase_feedback"])
    add_section("5. Spraak analyse", s["speech"])
    add_section("6. Gordon patterns", s["gordon"])
    add_section("7. Actiepunten", s["action_items"])
    add_section("8. Afsluiting", s["closing"])

    # Output the PDF to the specified filepath
    pdf.output(filepath)
