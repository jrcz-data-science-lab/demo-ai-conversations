from flask import Flask, request, jsonify
import requests
from db_utils import append_to_history, read_history, clear_history, store_audio_metadata, get_all_audio_metadata
from sqlite import init_db
from user_management import ensure_user
from speech_analysis import generate_speech_feedback
from gordon_patterns import generate_pattern_feedback
from feedback_formatter import format_student_feedback
from config import ENABLE_SPEECH_ANALYSIS
import os
import time
import logging
import base64

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

OLLAMA_URL = 'http://ollama:11434/api/generate'
TTS_URL = 'http://tts:5000/speech'
STT_URL = 'http://faster-whisper:5000/transcribe'
GENERATE_URL = 'http://127.0.0.1:8000/generate'
FEEDBACK_URL = 'http://127.0.0.1:8000/feedback'

def load_prompts(prompts_directory="prompts"):
    prompts = {}
    for filename in os.listdir(prompts_directory):
        if filename.endswith(".txt"):
            scenario_name = filename.split(".")[0]
            with open(os.path.join(prompts_directory, filename), "r", encoding="utf-8") as f:
                prompts[scenario_name] = f.read()
    return prompts

prompts = load_prompts() 

@app.route('/general', methods=['POST'])
def request_handling():
    data = request.json
    username = data.get("username")
    audio_in = data.get("audio")
    scenario = data.get("scenario")
    feedback_request = data.get("feedback", False)

    if scenario == '1':
        voice_model = "Kumar Dahl"
    elif scenario == '2':
        voice_model = "Luis Moray"
    elif scenario == '3':
        voice_model = "Wulf Carlevaro"
    elif scenario == '4':
        voice_model = "Filip Traverse"
    else:
        voice_model = "Damien Black"

    if not username or not audio_in or not scenario:
        return jsonify({"error": "Missing username, audio, or scenario"}), 400

    # Transcribe audio
    stt_resp = requests.post(STT_URL, json={"audio": audio_in})
    stt_json = stt_resp.json()
    transcription_text = stt_json.get("transcript", "")
    transcript_details = stt_json.get("transcript_details", {})
    
    # Calculate audio duration from audio data or get from transcript_details
    audio_duration = transcript_details.get("audio_duration", 0)
    if not audio_duration:
        # Fallback: estimate from audio size (rough approximation)
        try:
            audio_bytes = base64.b64decode(audio_in)
            # Assume 16-bit mono WAV at 48000 Hz (from test-requests.py)
            audio_duration = len(audio_bytes) / (48000 * 2)  # Rough estimate
        except:
            audio_duration = 0

    if not feedback_request:
        generate_resp = requests.post(GENERATE_URL, json={
            "username": username,
            "transcript": transcription_text,
            "scenario": scenario,
            "voice": voice_model,
            "transcript_details": transcript_details,
            "audio_duration": audio_duration
        })
        audio_b64 = generate_resp.json().get("audio")
        return jsonify({"audio": audio_b64})
    else:
        feedback_resp = requests.post(FEEDBACK_URL, json={
            "username": username,
            "scenario": scenario,
            "voice": voice_model
        })
        feedback_json = feedback_resp.json()
        # Return full feedback response including speech_metrics and icon_states
        return jsonify(feedback_json)


@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    username = data.get("username")
    transcript = data.get("transcript")
    scenario = data.get("scenario")
    voice = data.get("voice")
    transcript_details = data.get("transcript_details", {})
    audio_duration = data.get("audio_duration", 0)

    if not username or not transcript or not scenario:
        return jsonify({"error": "Missing username, transcript, or scenario"}), 400

    ensure_user(username)
    # Store message and get message_id
    message_id = append_to_history(username, "Student", transcript)
    
    # Store audio metadata for speech analysis if enabled
    if ENABLE_SPEECH_ANALYSIS and transcript_details:
        word_count = len(transcript.split())
        store_audio_metadata(username, message_id, audio_duration, transcript_details, word_count)
        print(f"[DEBUG] Stored audio metadata for user: {username}, message_id: {message_id}, duration: {audio_duration}s, words: {word_count}")
    elif ENABLE_SPEECH_ANALYSIS and not transcript_details:
        print(f"[DEBUG] Speech analysis enabled but no transcript_details provided. Skipping metadata storage.")
    
    convo = read_history(username)

    # Dynamically load the prompt from the dictionary
    prompt_text = prompts.get(f"conversation{scenario}", None)
    if not prompt_text:
        return jsonify({"error": f"No prompt found for scenario {scenario}"}), 400

    # Format the prompt with the conversation history
    prompt_text = prompt_text.format(convo=convo)
    print(prompt_text)

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json={"prompt": prompt_text, "model": "mistral-small3.2:24b", "stream": False, "think": False}
        )
        ollama_response.raise_for_status()
        response_text = ollama_response.json().get("response", "")
        print(response_text)

        if response_text:
            append_to_history(username, "Avatar", response_text)
            tts_resp = requests.post(TTS_URL, json={"text": response_text, "voice": voice})
            audio_b64 = tts_resp.json().get("audio")
            return jsonify({"response": response_text, "audio": audio_b64})

        return jsonify({"error": "Empty response from model"}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ollama error: {e}"}), 500

@app.route('/feedback', methods=['POST'])
def generate_feedback():
    data = request.json
    username = data.get("username")
    scenario = data.get("scenario")
    voice = data.get("voice")

    if not username or not scenario:
        return jsonify({"error": "Missing username or scenario"}), 400

    ensure_user(username)
    convo = read_history(username)

    # Dynamically load the feedback prompt from the dictionary
    feedback_prompt = prompts.get(f"feedback{scenario}", None)
    if not feedback_prompt:
        return jsonify({"error": f"No feedback prompt for scenario {scenario}"}), 400

    # Format the prompt with the conversation history
    feedback_prompt = feedback_prompt.format(convo=convo)

    try:
        # Generate conversation content feedback from Ollama
        ollama_response = requests.post(
            OLLAMA_URL,
            json={"prompt": feedback_prompt, "model": "qwen3:32b", "stream": False, "think": False}
        )
        ollama_response.raise_for_status()
        conversation_feedback = ollama_response.json().get("response", "")

        print("Conversation feedback:", conversation_feedback)
        
        # Generate Gordon pattern analysis feedback
        gordon_pattern_result = None
        print("[DEBUG] Starting Gordon pattern analysis...")
        print(f"[DEBUG] Conversation history length: {len(convo) if convo else 0} characters")
        try:
            gordon_pattern_result = generate_pattern_feedback(convo)
            print(f"[DEBUG] Gordon pattern analysis completed. Covered {gordon_pattern_result.get('covered_patterns', 0)}/11 patterns")
            print(f"[DEBUG] Gordon pattern summary: {gordon_pattern_result.get('summary', '')[:100]}...")
        except ImportError as e:
            logging.error(f"Gordon pattern analysis import failed: {e}")
            print(f"[ERROR] Gordon pattern analysis import failed: {e}")
            import traceback
            traceback.print_exc()
            gordon_pattern_result = None
        except Exception as e:
            logging.error(f"Gordon pattern analysis failed: {e}")
            print(f"[ERROR] Gordon pattern analysis exception: {e}")
            import traceback
            traceback.print_exc()
            gordon_pattern_result = None
        
        # Generate speech pattern feedback if enabled
        speech_analysis_result = None
        if ENABLE_SPEECH_ANALYSIS:
            try:
                # Retrieve audio metadata for analysis
                audio_metadata_list = get_all_audio_metadata(username)
                print(f"[DEBUG] Speech analysis enabled. Retrieved {len(audio_metadata_list)} audio metadata entries for user: {username}")
                
                if audio_metadata_list:
                    # Time the speech analysis (pass conversation history for content quality analysis)
                    start = time.time()
                    speech_analysis_result = generate_speech_feedback(audio_metadata_list, conversation_history=convo)
                    analysis_time = time.time() - start
                    logging.info(f"Speech analysis runtime: {analysis_time:.2f}s")
                    print(f"[DEBUG] Speech analysis completed. Metrics: {speech_analysis_result.get('metrics', {})}")
                else:
                    print(f"[DEBUG] No audio metadata found for user: {username}. Speech analysis skipped.")
            except Exception as e:
                logging.error(f"Speech analysis failed: {e}")
                print(f"[DEBUG] Speech analysis exception: {e}")
                import traceback
                traceback.print_exc()
                # Continue without speech analysis if it fails
                speech_analysis_result = None
        else:
            print("[DEBUG] Speech analysis is disabled (ENABLE_SPEECH_ANALYSIS = False)")
        
        # Format all feedback into student-friendly structure
        if conversation_feedback or gordon_pattern_result or speech_analysis_result:
            formatted_feedback = format_student_feedback(
                conversation_feedback,
                gordon_pattern_result,
                speech_analysis_result
            )
            print("[DEBUG] Feedback formatted successfully")
        else:
            fallback_text = conversation_feedback if conversation_feedback else "Geen feedback beschikbaar."
            formatted_feedback = {
                "text": fallback_text,
                "structured": {
                    "sections": {"summary": fallback_text},
                    "metadata": {}
                }
            }
        
        feedback_text = (formatted_feedback or {}).get("text")
        structured_feedback = (formatted_feedback or {}).get("structured", {})

        if feedback_text:
            tts_resp = requests.post(TTS_URL, json={"text": feedback_text, "voice": voice})
            audio_b64 = tts_resp.json().get("audio")

            # Prepare response
            response_data = {
                "response": feedback_text,
                "audio": audio_b64,
                "structured_feedback": structured_feedback
            }
            
            # Add speech metrics and icon states if available
            if speech_analysis_result:
                response_data["speech_metrics"] = speech_analysis_result.get("metrics", {})
                response_data["speech_summary"] = speech_analysis_result.get("summary", "")
                response_data["icon_states"] = speech_analysis_result.get("icon_states", {})
                response_data["confidence"] = speech_analysis_result.get("confidence", {})
                print(f"[DEBUG] Added speech metrics to response: {response_data.get('speech_metrics', {})}")
                print(f"[DEBUG] Added confidence score: {response_data.get('confidence', {})}")
                print(f"[DEBUG] Added icon states to response: {response_data.get('icon_states', {})}")
            else:
                print("[DEBUG] No speech_analysis_result available. Speech metrics not included in response.")
            
            # Add Gordon pattern analysis if available
            if gordon_pattern_result:
                response_data["gordon_patterns"] = {
                    "total_patterns": gordon_pattern_result.get("total_patterns", 11),
                    "covered_patterns": gordon_pattern_result.get("covered_patterns", 0),
                    "coverage_percentage": gordon_pattern_result.get("coverage_percentage", 0),
                    "mentioned_patterns": gordon_pattern_result.get("mentioned_patterns", []),
                    "summary": gordon_pattern_result.get("summary", "")
                }
                print(f"[DEBUG] Added Gordon pattern analysis to response: {response_data.get('gordon_patterns', {})}")
            else:
                print("[DEBUG] No gordon_pattern_result available. Gordon pattern analysis not included in response.")
            
            clear_history(username)

            return jsonify(response_data)

        return jsonify({"error": "Empty feedback response"}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ollama error: {e}"}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8000)
