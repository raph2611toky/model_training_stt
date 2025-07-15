from transformers import WhisperProcessor, WhisperForConditionalGeneration
import ffmpeg
import numpy as np
import jiwer
import os
import zipfile
import tempfile
import argparse

# Charger un modèle à partir d'un fichier zip
def load_model_from_zip(zip_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(tmpdirname)
        processor = WhisperProcessor.from_pretrained(tmpdirname)
        model = WhisperForConditionalGeneration.from_pretrained(tmpdirname)
    return processor, model

# Arguments pour spécifier le fichier zip
parser = argparse.ArgumentParser()
parser.add_argument("--model_zip", type=str, default="./whisper_malagasy_final", help="Chemin vers le fichier zip du modèle")
args = parser.parse_args()

# Charger le modèle
if os.path.isfile(args.model_zip) and args.model_zip.endswith(".zip"):
    processor, model = load_model_from_zip(args.model_zip)
else:
    processor = WhisperProcessor.from_pretrained(args.model_zip)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_zip)

def load_audio_with_ffmpeg(audio_path, target_sr=16000):
    try:
        out, _ = (
            ffmpeg.input(audio_path)
            .output("pipe:", format="f32le", acodec="pcm_f32le", ar=target_sr, ac=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.float32)

def test_audio(audio_path, reference_transcription=None):
    audio_array = load_audio_with_ffmpeg(audio_path)
    inputs = processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(inputs, max_length=100, language="mg")
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"Audio: {audio_path}")
    print(f"Transcription prédite: {transcription}")
    if reference_transcription:
        wer = jiwer.wer(reference_transcription, transcription)
        print(f"Word Error Rate (WER): {wer:.4f}")

if __name__ == "__main__":
    test_audio("./datasets/audios/izaho/1.m4a")