import json
import requests

def write_email(data: dict) -> str:
    OLLAMA_URL = "http://ollama:11434/api/generate"
    # Serialize data to JSON string for better readability in prompt
    data_json = json.dumps(data, indent=2)

    prompt = f"""
    Je bent een feedbackcoach die constructieve feedback geeft aan een student. De student is beoordeeld op basis van de volgende gegevens:

    {data_json}

    Op basis van deze data schrijf je een e-mail aan de student in drie paragrafen:

    1. In de eerste paragraaf, leg uit wat de student goed heeft gedaan, gebruik maximaal 3 zinnen.
    2. In de tweede paragraaf, leg uit wat de student kan verbeteren, gebruik maximaal 3 zinnen.
    3. In de derde paragraaf, geef advies over hoe de student zich kan verbeteren gebruik maximaal 3 zinnen.

    De toon moet constructief en bemoedigend zijn. Eindig de e-mail met het volgende:
    - Wens de student veel succes en moedig hem/haar aan.
    - Sluit af met "Talk2Care Project" als ondertekening.

Let op:
- Gebruik **geen Markdown-formatering** of speciale tekens zoals sterretjes (***), underscores (_), of andere symbolen die op opmaak wijzen. Dit is een gewone platte tekst-e-mail.
- Zorg ervoor dat je duidelijk en beknopt bent, en vermijd het gebruik van de naam van de student of persoonlijke identificatiegegevens. Verwijs alleen naar de student als "de student."
    """

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json={"prompt": prompt, "model": "qwen3:32b", "stream": False, "think": False}
        )
        ollama_response.raise_for_status()
        conversation_feedback = ollama_response.json().get("response", "")
        
        return conversation_feedback
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Ollama error: {e}"}, 500