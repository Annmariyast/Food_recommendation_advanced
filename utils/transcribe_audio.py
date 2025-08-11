# utils/transcribe_audio.py
import os
import io
import speech_recognition as sr
from pydub import AudioSegment

def transcribe_audio_input(uploaded_file_or_bytes):
    """Transcribe audio from file upload or raw bytes.

    Strategy:
    1) Try reading bytes and passing directly to SpeechRecognition's AudioFile (works for WAV/AIFF/FLAC).
    2) Fallback to pydub (requires ffmpeg) to convert to WAV, then transcribe.
    """
    try:
        # Normalize to raw bytes
        data: bytes
        source_obj = uploaded_file_or_bytes
        if isinstance(source_obj, (bytes, bytearray)):
            data = bytes(source_obj)
        elif hasattr(source_obj, 'read'):
            data = source_obj.read()
        else:
            # assume file path
            with open(source_obj, 'rb') as f:
                data = f.read()

        recognizer = sr.Recognizer()

        # First attempt: direct decode (no ffmpeg dependency if WAV/AIFF/FLAC)
        try:
            with sr.AudioFile(io.BytesIO(data)) as source:
                audio_data = recognizer.record(source)
                return recognizer.recognize_google(audio_data)
        except Exception:
            # Fallback: use pydub to transcode to WAV
            temp_path = "temp_audio.wav"
            audio = AudioSegment.from_file(io.BytesIO(data))
            audio.export(temp_path, format="wav")
            with sr.AudioFile(temp_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
            os.remove(temp_path)
            return text
    except Exception as e:
        print("Audio transcription error:", e)
        return ""
