#!/bin/sh

CONVERSATION_MODEL="phi4:latest"
FEEDBACK_MODEL="qwen3:32b"

pip install -r requirements.txt --break-system-packages

ollama serve &

until ollama list >/dev/null 2>&1; do
  echo "Waiting for ollama to become available..."
  sleep 1
done

if ! ollama list | grep -q "$CONVERSATION_MODEL"; then
  echo "Model '$CONVERSATION_MODEL' not found locally. Pulling..."
  ollama pull "$CONVERSATION_MODEL"
else
  echo "Model '$CONVERSATION_MODEL' already present."
fi

if ! ollama list | grep -q "$FEEDBACK_MODEL"; then
  echo "Model '$FEEDBACK_MODEL' not found locally. Pulling..."
  ollama pull "$FEEDBACK_MODEL"
else
  echo "Model '$FEEDBACK_MODEL' already present."
fi

wait
