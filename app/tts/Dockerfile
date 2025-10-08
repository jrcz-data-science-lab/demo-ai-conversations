FROM python:3.10

WORKDIR /usr/src/app

RUN apt update -y && apt install -y alsa-utils

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

ENTRYPOINT ["/bin/bash", "script.sh"]
