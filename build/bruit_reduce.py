import os
import noisereduce as nr
import librosa
import numpy as np
from pydub import AudioSegment
from utils.gotique import make_gotique

audio_dir = "./datasets/audios"

def reduce_noise_in_place(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    
    noise_sample = audio[0:int(0.5 * sr)]
    
    reduced_audio = nr.reduce_noise(y=audio, y_noise=noise_sample, sr=sr)
    
    reduced_audio_int = (reduced_audio * 32767).astype(np.int16)
    
    audio_segment = AudioSegment(
        reduced_audio_int.tobytes(),
        frame_rate=sr,
        sample_width=2, 
        channels=1 
    )
    
    audio_segment.export(audio_path, format="mp4")

def process_all_audio_files(audio_dir="./datasets/audios"):
    print(f"\n{make_gotique('- Bruit ...')}")
    
    for subdir in os.listdir(audio_dir):
        subdir_path = os.path.join(audio_dir, subdir)
        if os.path.isdir(subdir_path):
            audio_files = [f for f in os.listdir(subdir_path) if f.endswith(".m4a")]
            for audio_file in audio_files:
                audio_path = os.path.join(subdir_path, audio_file)
                print(f"Traitement de {audio_path}")
                reduce_noise_in_place(audio_path)
                print(f"Fichier nettoyé : {audio_path}")
