FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN apt update -y && apt install -y alsa-utils

COPY . .

EXPOSE 5000

ENTRYPOINT ["/bin/bash", "script.sh"]
