FROM ollama/ollama:0.13.1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 11434 5000

ENTRYPOINT ["/app/entrypoint.sh"]
