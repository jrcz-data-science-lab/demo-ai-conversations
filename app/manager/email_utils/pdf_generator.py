from fpdf import FPDF
from pathlib import Path

def create_feedback_pdf(data: dict, filepath: str, logo_path: str | None = None):
    # Create PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = Path("/usr/src/app/email_utils/dejavu-sans/DejaVuSans.ttf")

    # Check if the font file exists before proceeding
    if not font_path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")

    # Add only the regular font
    pdf.add_font("dejavu-sans", style="", fname=str(font_path))
    
    # Set the font for the document content (regular style)
    pdf.set_font("dejavu-sans", style="", size=11)  # Use regular font

    s = data["structured"]["sections"]

    def add_section(title, text):
        # Set regular font for section titles and content
        pdf.set_font("dejavu-sans", style="", size=12)  # Regular font for section titles
        pdf.multi_cell(0, 8, title)
        pdf.ln(1)
        
        # Use regular font for section content
        pdf.set_font("dejavu-sans", style="", size=11)  # Regular font for content
        pdf.multi_cell(0, 7, text)
        pdf.ln(4)

    # Add all sections
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
