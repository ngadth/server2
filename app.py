import os
import requests
import base64
import time
import librosa
from flask import Flask, redirect, render_template, request, session, jsonify, send_file
from flask_socketio import SocketIO, send, emit, join_room, leave_room
# from scipy.io.wavfile import write as write_wav
from loguru import logger
from OpenSSL import SSL
from moviepy.editor import *
from infer import VietASR


app = Flask(__name__)
app.config["SECRET_KEY"] = "dangvansam"
socketio = SocketIO(app)

config = 'configs/quartznet12x1_vi.yaml'
encoder_checkpoint = 'models/acoustic_model/vietnamese/JasperEncoder-STEP-289936.pt'
decoder_checkpoint = 'models/acoustic_model/vietnamese/JasperDecoderForCTC-STEP-289936.pt'
lm_path = 'models/language_model/3-gram-lm.binary'

vietasr = VietASR(
    config_file=config,
    encoder_checkpoint=encoder_checkpoint,
    decoder_checkpoint=decoder_checkpoint,
    lm_path=lm_path,
    beam_width=50
)



STATIC_DIR = "static"
UPLOAD_DIR = "upload"
RECORD_DIR = "record"
VIDEO_DIR = "video"

os.makedirs(os.path.join(STATIC_DIR, UPLOAD_DIR), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, RECORD_DIR), exist_ok=True)

@app.route("/")
def index():
    return render_template(
        template_name_or_list="index.html",
        audio_path=None,
        async_mode=socketio.async_mode
    )


@socketio.on('connect')
def connected():
    logger.info("CONNECTED: " + request.sid)
    emit('to_client', {'text': request.sid})


@socketio.on('to_server')
def response_to_client(data):
    logger.info(data["text"])
    emit('to_client', {'text': len(data["text"].split())})


@socketio.on('audio_to_server')
def handle_audio_from_client(data):
    filename = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(STATIC_DIR, RECORD_DIR, filename + ".wav")
    audio_file = open(filepath, "wb")
    decode_string = base64.b64decode(data["audio_base64"].split(",")[1])
    audio_file.write(decode_string)
    logger.info("asr processing...")
    audio_signal, _ = librosa.load(filepath, sr=16000)
    transcript = vietasr.transcribe(audio_signal)
    logger.success(f'transcript: {transcript}')
    emit('audio_to_client', {'filepath': filepath, 'transcript': transcript})


# def convert_mp4_to_wav(input_file, output_file):
#     video_clip = VideoFileClip(input_file)
#     audio_clip = video_clip.audio
#     audio_clip.write_audiofile(output_file)
#     audio_clip.close()

# def convert_mp4_to_wav(input_file, output_file):
#     for filename in os.listdir(input_file):
#         actual_filename = filename[:-4]
#         if(filename.endswith(".mp4")):
#             os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(filename, actual_filename))
#         else:
#             continue

def encode_base64():
    org_id = '4b54205f-a32d-4e1b-b4c5-cd6e1780a364'
    api_key = '3c2b56c6f9e6caecf5c1'

    if not org_id or not api_key:
        print('Environment variables ORG_ID and API_KEY are required')
        return None

    combined = f"{org_id}:{api_key}"
    encoded_bytes = base64.b64encode(combined.encode('utf-8'))
    encoded_string = encoded_bytes.decode('utf-8')

    return f"Basic {encoded_string}"

def call_api_audio(recording_id):
    api_url = f"https://api.dyte.io/v2/recordings/{recording_id}"
    
    headers = {
        "Authorization": encode_base64(),  # Add your authorization token here
        "Content-Type": "application/json"
        # Add any other headers you want to include here
    }

    response = requests.get(api_url, headers=headers)
    return response.json()

def download_video(url, record_id):
    VIDEO_FILE_TO_SAVE  = os.path.join(STATIC_DIR, VIDEO_DIR, record_id + ".mp4")  # Specify the full path where you want to save the video file
    AUDIO_FILE_TO_SAVE = os.path.join(STATIC_DIR, RECORD_DIR, record_id + ".wav")
    # Download the video file
    resp = requests.get(url)
    with open(VIDEO_FILE_TO_SAVE, "wb") as f:
        f.write(resp.content)
    # convert mp4 -> wav
    try:
        video_clip = VideoFileClip(VIDEO_FILE_TO_SAVE)
        video_clip.audio.write_audiofile(AUDIO_FILE_TO_SAVE)
        video_clip.close()
        return jsonify({"status": "success", "message": "Video converted to WAV successfully", "audio_file_path": AUDIO_FILE_TO_SAVE})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/upload', methods=['POST', 'GET'])
def handle_upload():
    if request.method == "POST":
        _file = request.files['file']
        if _file.filename == '':
            return index()
        print(f'file uploaded: {_file.filename}')
        filepath = os.path.join(STATIC_DIR, UPLOAD_DIR, _file.filename)
        _file.save(filepath)
        print(f'saved file to: {filepath}')
        audio_signal, _ = librosa.load(filepath, sr=16000)
        transcript = vietasr.transcribe(audio_signal)
        print(f'audio {audio_signal}')
        print(f'transcript: {transcript}')
        return render_template(
            template_name_or_list='index.html',
            transcript=transcript,
            audiopath=filepath
        )
        return
    else:
        return redirect("/")

@app.route('/translate', methods=['GET', 'POST'])
def handle_translate():
    if request.method == 'GET':
        id_param = request.args.get('id')
        if id_param:
            res = call_api_audio(id_param)
            if res.get("data", {}).get("data"):
                return jsonify({"error": "Data not found or invalid"}), 400

            id_record = res["data"]["id"]
            url_download = res["data"]["download_url"]
          
            download_video(url_download, id_record)

            filepath = os.path.join(STATIC_DIR, RECORD_DIR, id_record + ".wav")
            if(filepath):
                print(f'saved file to: {filepath}')
                audio_signal, _ = librosa.load(filepath, sr=16000)
                transcript = vietasr.transcribe(audio_signal)
                print(f'audio {audio_signal}')
                print(f'transcript: {transcript}')
                # fileName = id_param+'txt'
                # with open(fileName, 'w') as file:
                #     file.write(transcript)
                return jsonify({"data": transcript})
                # return send_file(fileName, as_attachment=True)
        else:
            return jsonify({"error": "No 'id' parameter provided in the GET request"}), 400
    else:
        return jsonify({"error": "Unsupported HTTP method"}), 405

@app.route('/download-video-transcript', methods=['GET', 'POST'])
def handle_download_transcript():
    if request.method == 'GET':
        id_param = request.args.get('id')
        if not id_param:
            return jsonify({"error": "ID parameter is missing"}), 400

        res = call_api_audio(id_param)
        id_record = res["data"]["id"]
        url_download = res["data"]["download_url"]
        print('Url download', url_download)

        return jsonify({"data": url_download})
           

if __name__ == '__main__':
    socketio.run(app, host="localhost", port=5000,
                 debug=True)
