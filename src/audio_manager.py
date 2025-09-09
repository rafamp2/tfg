import os
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
from mutagen.mp3 import MP3

import torch

# Para XTTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Para faster-whisper 
from faster_whisper import WhisperModel

from src import CFG


class AudioManagerXTTS:
    def __init__(self, language="es", speaker_wav_path=None):
        self.language = language
        self.speaker_wav = speaker_wav_path or os.path.join(
            CFG.AUDIO_CONFIG.AUDIO_DIR, CFG.AUDIO_CONFIG.VOICE_FILE
        )
        if not os.path.exists(self.speaker_wav):
            raise FileNotFoundError(f"El archivo '{self.speaker_wav}' no fue encontrado.")

        self.device = "cuda" if (CFG.TTS_DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"

        # Inicializar el modelo TTS 
        xtts_dir = os.path.join(CFG.MODELS_DIR, CFG.TTS_MODEL) 
        tts_config_path = os.path.join(xtts_dir, "config.json") 
        self.tts_config = XttsConfig() 
        self.tts_config.load_json(tts_config_path) 
        self.tts_model = Xtts.init_from_config(self.tts_config) 
        self.tts_model.load_checkpoint(self.tts_config, checkpoint_dir=xtts_dir, eval=True) 
        if self.device == "cuda":
            self.tts_model.cuda()

        # Inicializamos el modelo Whisper
        self.whisper_model = WhisperModel(CFG.WHISPER_MODEL, compute_type="auto")

        # ---------- Audio I/O ----------
        pygame.mixer.init(frequency=48000, buffer=1024)

        # Parámetros de grabación (mic)
        self.SAMPLE_RATE = 16000
        self.BLOCK_DURATION = 0.5
        self.SILENCE_THRESHOLD = 0.15
        self.MAX_SILENT_BLOCKS = 4


    # ----------------- Calibración de silencio -----------------
    def calibrate(self, segundos=2):
        print(f"🎧 Calibrando ruido ambiental durante {segundos}s...")
        samples = []

        def _cb(indata, frames, time_info, status):
            samples.append(np.mean(np.abs(indata)))

        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, callback=_cb):
            time.sleep(segundos)

        ruido_medio = np.median(samples) if samples else self.SILENCE_THRESHOLD
        self.SILENCE_THRESHOLD = max(ruido_medio * 1.5, 1e-3)
        print(f"📊 Umbral auto: {self.SILENCE_THRESHOLD:.5f}")

    # ----------------- Grabación con VAD simple -----------------
    def record_callback(self, min_record_seconds=1.5):
        print("🎤 Starting recording. Speak now...")

        audio_q = queue.Queue()
        silent_blocks = 0
        recording = []
        start_time = time.time()

        def _cb(indata, frames, time_info, status):
            vol = float(np.mean(np.abs(indata)))
            audio_q.put(indata.copy())
            nonlocal silent_blocks
            silent_blocks = silent_blocks + 1 if vol < self.SILENCE_THRESHOLD else 0

        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1,
                            callback=_cb,
                            blocksize=int(self.SAMPLE_RATE * self.BLOCK_DURATION)):
            while True:
                block = audio_q.get()
                recording.append(block)
                elapsed = time.time() - start_time
                if elapsed >= min_record_seconds and silent_blocks >= self.MAX_SILENT_BLOCKS:
                    break

        print("🛑 Silence detected. Ending recording.")
        return np.concatenate(recording, axis=0).astype(np.float32).flatten()

    # ----------------- Transcripción (Whisper HF) -----------------
    def transcribe_audio(self, audio_np_array: np.ndarray):
        print("📝 Transcribing...")

        # Asegurarse de que el audio sea float32
        if audio_np_array.dtype != np.float32:
            audio_np_array = audio_np_array.astype(np.float32)

        # Usamos el modelo Whisper
        segments, _ = self.whisper_model.transcribe(
            audio=audio_np_array,
            language=self.language,
            beam_size=5,
            vad_filter=True
        )

        # Concatenar todos los segmentos en un solo texto 
        text = " ".join([segment.text for segment in segments]).strip() 
        return text

    def listen(self):
        audio = self.record_callback()
        return self.transcribe_audio(audio)

    # ----------------- Síntesis (XTTS v2) -----------------
    def speak(self, text: str) -> None:
        """
        Genera audio con XTTS v2 local y lo reproduce.

        Args:
            text (str): Texto a convertir en audio.
        """
        print("Generating audio...")
        voice_style_path = os.path.join(CFG.AUDIO_CONFIG.AUDIO_DIR,CFG.AUDIO_CONFIG.VOICE_FILE)
        # Generar audio con XTTS
        outputs = self.tts_model.synthesize(
            text,
            self.tts_config,
            speaker_wav=voice_style_path,
            gpt_cond_len=CFG.AUDIO_CONFIG.GPT_COND_LEN,
            language=self.language,
        )

        # Extraer el audio generado
        waveform = outputs["wav"]

        # Obtener sample rate desde la configuración
        sr = self.tts_config.audio.sample_rate

        # Reproducir el audio directamente
        sd.play(waveform, sr)
        sd.wait()