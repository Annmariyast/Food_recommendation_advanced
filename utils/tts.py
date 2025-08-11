from io import BytesIO
from typing import Optional, Tuple
import os
import sys
import tempfile
import subprocess
import shutil

from gtts import gTTS
from pydub import AudioSegment


def _synthesize_gtts(text: str, lang: str = "en") -> Optional[Tuple[bytes, str]]:
    try:
        tts = gTTS(text=text, lang=lang)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read(), "audio/mp3"
    except Exception:
        return None


def _synthesize_macos_say(text: str) -> Optional[Tuple[bytes, str]]:
    if sys.platform != "darwin":
        return None
    if not shutil.which("say"):
        return None
    try:
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "tts.aiff")
            # 22.05 kHz PCM little endian, works well in browsers
            subprocess.run([
                "say", "-o", out_path, "--data-format=LEI16@22050", text
            ], check=True)
            # Convert AIFF -> WAV for broad browser support
            with open(out_path, "rb") as f:
                aiff_bytes = f.read()
            seg = AudioSegment.from_file(BytesIO(aiff_bytes), format="aiff")
            wav_buf = BytesIO()
            seg.export(wav_buf, format="wav")
            wav_buf.seek(0)
            return wav_buf.read(), "audio/wav"
    except Exception:
        return None


def synthesize_speech_to_bytes(text: str, lang: str = "en") -> Optional[Tuple[bytes, str]]:
    """Generate speech audio bytes from text.

    Tries online gTTS (MP3). If it fails, on macOS falls back to the local
    `say` command producing AIFF.
    """
    text = (text or "").strip()
    if not text:
        return None
    audio = _synthesize_gtts(text, lang=lang)
    if audio:
        return audio
    return _synthesize_macos_say(text)

