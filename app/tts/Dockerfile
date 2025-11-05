FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /usr/src/app

RUN apt update -y && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN git clone https://github.com/idiap/coqui-ai-TTS /tmp/coqui-ai-TTS && \
    cd /tmp/coqui-ai-TTS && \
    pip install -e .

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "tts.py"]