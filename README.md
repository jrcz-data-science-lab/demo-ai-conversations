# Demo-AI-Conversation
This repository contains the backend code for the **Cornelius AI Project: Talk2Care**. Talk2Care is designed to be used for research in an educational settings, too see how well AI can be used as an educational tool. For this case it is meant too simulate real-world medical scenarios. The AI acts as a virtual patient, allowing users to practice their communication skills before engaging with real patients.

While the backend is primarily designed to work with a frontend, it can also function independently. You can interact with the backend directly using the `client.py` file, which allows you to send requests to the backend for processing.

## System Requirements
The system requirements may vary depending on the models being used. Below are the **minimum requirements** necessary to run the project:

- **Docker**: Required to containerize the backend services.
- **Python version 3 or later**: The project is built with Python 3.
- **16 GB RAM (minimum)**: Docker and AI models have a lot of memory demand.
- **Nvidia GPU (optional but recommended)**: The models used in development are boosted by the GPU. However if Nvidia is not available it will default to CPU.

### Operating Systems:
- **Linux** (recommended)
- **macOS**
- **Windows** (with Docker Desktop)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jrcz-data-science-lab/demo-ai-conversation.git
   cd demo-ai-conversation
   ```

2. **Build the backend using Docker:**

   ```bash
   docker compose up --build
   ```

3. **[OPTIONAL] Run the client:**

   ```bash
   python client.py
   ```

**Note:** The final build includes large models. If you prefer a lightweight setup, it's recommended to replace the models before running the Docker commands.

Once the backend is running locally, you can access it at `http://localhost:8000/general` this is the main entry point. If you are only working with the backend, you can use the `client.py` file to send requests to the backend.

review my paragraph of my readme pls

## Changing Models
### Speech-To-Text
STT Component in ``faster-whisper.py`` file on line 11 you can change the ``large-turbo-v3`` this specifies the models for faster whisper refer to the repository for a list of models and there specifications: https://github.com/SYSTRAN/faster-whisper

```
model = WhisperModel("large-v3-turbo", compute_type="int8")
```

### Ollama
Ollama Component you will need to change two files in Ollama's folder go to the ``entrypoint.sh`` file and change the variable in line 3 or 4 for the models that you like to use:

```
CONVERSATION_MODEL="gemma3:27b"
FEEDBACK_MODEL="qwen3:32b"
```

These two models have different tasks. If you change the conversation model you need to go to the manager folder and go to app.py in line 163 you will need to change the model in the parameters here too:

```
ollama_response = requests.post(
  OLLAMA_URL,
  json={"prompt": prompt_text, "model": "gemma3:27b", "stream": False, "think": False}
)
```

If you change the feedback model you have to go the same file and instead go to line 214 change the parameter of the model you like to use:
```
ollama_response = requests.post(
  OLLAMA_URL,
  json={"prompt": feedback_prompt, "model": "qwen3:32b", "stream": False, "think": False}
)
```
For a list of models you can use the Ollama library provided in this link: https://ollama.com/library

### Text-To-Speech
TTS Component in ``tts.py`` at line 22 you can change ``xtts_v2`` to the model you like to use here: https://github.com/coqui-ai/TTS

```
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
```