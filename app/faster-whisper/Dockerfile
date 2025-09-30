#FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    libportaudio2 \
    libsndfile1 \
    alsa-utils \
    pulseaudio-utils \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "faster-whisper.py"]
