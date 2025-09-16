# demo-ai-conversations

# Ollama

- Build
  ''' docker build -t ollama . '''
  
- Start
  ''' docker run --rm -it -v $(pwd):/app -p 11434:11434 -p 8000:8000 --network mynet --name ollama ollama '''


# Piper TTS

- Build
  ''' docker build -t tts . '''

- Start
  ''' docker run --rm -itd -v $(pwd):/usr/src/app --network mynet -p 5000:5000 --name tts tts '''


