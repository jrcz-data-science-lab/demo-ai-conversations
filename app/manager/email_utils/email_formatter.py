import json

def format_feedback_email(data: dict) -> str:
    # If 'data' is a string, attempt to parse it as JSON
    if isinstance(data, str):
        try:
            data = json.loads(data)  # Convert the string to a dictionary
        except json.JSONDecodeError:
            raise ValueError("The provided string could not be parsed into JSON.")

    # Check if data is a dictionary and contains the expected keys
    if not isinstance(data, dict):
        raise TypeError("Expected 'data' to be a dictionary, but got {0}".format(type(data)))

    # Ensure the expected structure is present
    if "structured" not in data or "sections" not in data["structured"]:
        raise KeyError("'structured' or 'sections' not found in the provided data.")

    # Safely extract 'sections' from 'data'
    s = data["structured"]["sections"]

    return f"""
Beste student,

Bedankt voor je deelneming aan onze sessie. In deze email staan je resultaten op basis van het gesprek.

{s["summary"]}

{s["gespreksvaardigheden"]}

{s["comprehension"]}

{s["phase_feedback"]}

{s["speech"]}

{s["gordon"]}

{s["action_items"]}

{s["closing"]}

Met vriendelijke groeten,
Het Talk2Care Team
""".strip()