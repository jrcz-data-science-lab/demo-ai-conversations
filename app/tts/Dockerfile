FROM ubuntu:22.04

WORKDIR /usr/src/app

# Install system dependencies
RUN apt update -y && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Install Coqui TTS from official repository
RUN pip install TTS

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "tts.py"]
