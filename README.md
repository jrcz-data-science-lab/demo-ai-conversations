# demo-ai-conversations




# Ollama
- Post requests: 
  I made a python request handler that will pick up and handle post requests made to port 8000 of the server. Whenever a request is done it will feed that request to Ollama, take the response, and send it to Piper to generate audio output.

  An example of how you can build a simple post request:
  ```bash
  curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hoe gaat het?"}'
  ```
- Build
  ```bash
  docker build -t ollama .
  ```
- Run
  ```bash
  docker run --rm -it -v $(pwd):/app -p 11434:11434 -p 8000:8000 --network mynet --name ollama ollama
  ```


# Piper TTS

- Build
  ```bash
  docker build -t tts .
  ```
- Run
  ```bash
  docker run --rm -itd -v $(pwd):/usr/src/app --network mynet -p 5000:5000 --name tts tts
  ```
# Faster-Whisper
- Build
  ```bash
  docker build -t tts .
  ```
- Run
  ```bash
  docker run -it -v $(pwd):/app --device /dev/snd:/dev/snd --network mynet --name faster-whisper faster-whisper
  ```

