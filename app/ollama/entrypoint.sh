#!/bin/sh

MODEL="qwen3:0.6b"

pip install -r requirements.txt --break-system-packages

ollama serve &

until ollama list >/dev/null 2>&1; do
  echo "Waiting for ollama to become available..."
  sleep 1
done

if ! ollama list | grep -q "$MODEL"; then
  echo "Model '$MODEL' not found locally. Pulling..."
  ollama pull "$MODEL"
else
  echo "Model '$MODEL' already present."
fi

wait
