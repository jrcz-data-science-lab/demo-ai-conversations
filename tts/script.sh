#!/bin/bash
cd /usr/src/app
python -m piper.download_voices nl_NL-ronnie-medium
flask --app tts run --host=0.0.0.0