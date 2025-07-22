import torch
import sounddevice as sd
from mutagen.mp3 import MP3
import soundfile as sf
import os
from src import CFG
import pygame
import time
import numpy as np

# Para faster-whisper
from faster_whisper import WhisperModel

# Para XTTS
#from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

# Registrar clases seguras
#add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])


def build_tts():
    
    tts = TTS(
        model_path= os.path.join(CFG.MODELS_DIR, CFG.TTS_PATH),
        config_path=os.path.join(CFG.MODELS_DIR, CFG.TTS_CONFIG),
    )
    tts.to(CFG.DEVICE)
    return tts

class AudioManagerXTTS:
    def __init__(self, tts, language="es", speaker_wav_path = os.path.join(CFG.AUDIO_CONFIG.AUDIO_DIR, CFG.AUDIO_CONFIG.VOICE_FILE)):
        # Validar existencia del archivo de voz        
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"El archivo '{speaker_wav_path}' no fue encontrado.")

        self.speaker_wav = speaker_wav_path
        self.language = language

        self.tts = tts

        pygame.mixer.init(frequency=48000, buffer=1024)

        # Inicializar pygame para reproducir
        pygame.mixer.init(frequency=48000, buffer=1024)

        # Inicializar Whisper para transcripción
        self.whisper_model = WhisperModel("medium", compute_type="auto")

        # Parámetros de grabación
        self.SAMPLE_RATE = 16000
        self.BLOCK_DURATION = 0.5
        self.SILENCE_THRESHOLD = 0.15
        self.MAX_SILENT_BLOCKS = 4

    def play_audio(self, audios_dir, sleep_during_playback=True, delete_file=False, play_using_music=True):
        """
        Parameters:
        file_path (str): path to the audio file
        sleep_during_playback (bool): means program will wait for length of audio file before returning
        delete_file (bool): means file is deleted after playback (note that this shouldn't be used for multithreaded function calls)
        play_using_music (bool): means it will use Pygame Music, if false then uses pygame Sound instead
        """
        file_path = os.path.join(audios_dir, CFG.AUDIO_CONFIG.VOICE_FILE)
        print(f"Reproduciendo audio: {file_path}")
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=48000, buffer=1024)

        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.wav':
            sound = pygame.mixer.Sound(file_path)
            sound.play()
            duration = sf.SoundFile(file_path).frames / sf.SoundFile(file_path).samplerate
        elif ext.lower() == '.mp3':
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            duration = MP3(file_path).info.length
        else:
            print("Formato de audio no soportado.")
            return

        time.sleep(duration)

        if delete_file:
            try:
                os.remove(file_path)
                print("Archivo de audio eliminado.")
            except Exception as e:
                print(f"No se pudo eliminar el archivo: {e}")

    def calibrate(self, segundos=2):
        print(f"🎧 Calibrando ruido ambiente durante {segundos}s...")
        samples = []

        def callback(indata, frames, time_info, status):
            volume = np.mean(np.abs(indata))
            samples.append(volume)

        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, callback=callback):
            time.sleep(segundos)

        ruido_medio = np.mean(samples)
        self.SILENCE_THRESHOLD = ruido_medio * 1.5  # un poco por encima del ruido
        print(f"📊 Umbral ajustado automáticamente: {self.SILENCE_THRESHOLD:.5f}")

    def record_callback(self):
        print("🎤 Comenzando grabación. Habla cuando quieras...")
        audio_q = queue.Queue()
        silent_blocks = 0
        recording = []

        def callback(indata, frames, time_info, status):
            volume_norm = np.mean(np.abs(indata))
            print(f"Volumen: {volume_norm:.5f}")
            audio_q.put(indata.copy())
            nonlocal silent_blocks
            if volume_norm < self.SILENCE_THRESHOLD:
                silent_blocks += 1
            else:
                silent_blocks = 0

        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1,
                            callback=callback,
                            blocksize=int(self.SAMPLE_RATE * self.BLOCK_DURATION)):
            while True:
                block = audio_q.get()
                recording.append(block)
                if silent_blocks >= self.MAX_SILENT_BLOCKS:
                    break

        print("🛑 Silencio detectado. Finalizando grabación.")
        return np.concatenate(recording).flatten()

    def transcribe_audio(self, audio_np_array):
        print("📝 Transcribiendo...")
        segments, _ = self.whisper_model.transcribe(audio_np_array, language=self.language)
        full_text = ""
        for segment in segments:
            print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")
            full_text += segment.text.strip() + " "
        return full_text.strip()

    def listen(self):
        audio = record_callback()
        return transcribe_audio(audio)

    def sintetize_text(self, text, output_path="voz_generada.wav"):
        print("🧠 Sintetizando texto...")
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=self.speaker_wav,
            language=self.language
        )
        return output_path

    def speak(self, text, audio_dir, filename = "voz_generada.wav", delete_after_play=True):
        output_path = os.path.join(audio_dir,filename)
        # Generar el audio con XTTS
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=self.speaker_wav,
            language=self.language
        )
        # Reproducir y eliminar si se desea
        self.play_audio(audio_dir, delete_file=delete_after_play)
