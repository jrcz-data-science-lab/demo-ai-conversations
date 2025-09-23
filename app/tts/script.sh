#!/bin/bash
cd /usr/src/app
echo -e 'pcm.!default {\n  type plug\n  slave.pcm "dmix"\n}' > ~/.asoundrc
python -m piper.download_voices nl_NL-ronnie-medium
flask --app tts run --host=0.0.0.0