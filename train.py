from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset
from transformers import Trainer, TrainingArguments
import jiwer
import os
import ffmpeg
import numpy as np
import torch
import random
import string
import datetime
import zipfile
import tempfile
import argparse

audio_dir = "./datasets/audios"
transcription_dir = "./datasets/transcriptions"

def generate_zip_name(prefix="model"):
    letters = string.ascii_letters
    random_letters = ''.join(random.choice(letters) for _ in range(8))
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    return f"{prefix}_{random_letters}_{timestamp}.zip"

def zip_model(model_dir, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(model_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), model_dir))

def load_model_from_zip(zip_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(tmpdirname)
        processor = WhisperProcessor.from_pretrained(tmpdirname)
        model = WhisperForConditionalGeneration.from_pretrained(tmpdirname)
    return processor, model

parser = argparse.ArgumentParser()
parser.add_argument("--continue_from", type=str, default=None, help="Chemin vers le fichier zip du modèle pré-entraîné")
args = parser.parse_args()

if args.continue_from:
    processor, model = load_model_from_zip(args.continue_from)
else:
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

def load_custom_dataset(audio_dir, transcription_dir):
    data = {"audio": [], "transcription": []}
    for subdir in os.listdir(audio_dir):
        subdir_path = os.path.join(audio_dir, subdir)
        if os.path.isdir(subdir_path):
            transcription_file = f"{subdir}.txt"
            transcription_path = os.path.join(transcription_dir, transcription_file)
            if os.path.exists(transcription_path):
                with open(transcription_path, "r", encoding="utf-8") as f:
                    transcription = f.read().strip()
                audio_files = [f for f in os.listdir(subdir_path) if f.endswith(".m4a")]
                for audio_file in audio_files:
                    audio_path = os.path.join(subdir_path, audio_file)
                    data["audio"].append(audio_path)
                    data["transcription"].append(transcription)
    return Dataset.from_dict(data)

dataset = load_custom_dataset(audio_dir, transcription_dir)

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

def preprocess(batch):
    audio_array = load_audio_with_ffmpeg(batch["audio"])
    input_features = processor.feature_extractor(audio_array, sampling_rate=16000).input_features[0]
    batch["input_features"] = input_features
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

dataset = dataset.map(preprocess, remove_columns=["audio", "transcription"])
split = dataset.train_test_split(test_size=0.1)
train_dataset, eval_dataset = split["train"], split["test"]

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

training_args = TrainingArguments(
    output_dir="./whisper_malagasy",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    dataloader_pin_memory=False,
    num_train_epochs=5,
    gradient_accumulation_steps=2,
    fp16=True,
    save_strategy="epoch",
    logging_steps=50,
)

def compute_metrics(pred):
    pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    if pred_ids.ndim == 3:
        pred_ids = pred_ids.argmax(axis=-1)
    label_ids = pred.label_ids
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = jiwer.wer(label_str, pred_str)
    return {"wer": wer}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./whisper_malagasy_final")
    processor.save_pretrained("./whisper_malagasy_final")
    
    # Évaluation finale
    eval_results = trainer.evaluate()
    wer = eval_results["eval_wer"]
    print(f"WER final : {wer:.4f}")
    
    # Sauvegarde selon le WER
    if wer < 0.2:  # Seuil pour "utilisable"
        os.makedirs("./models", exist_ok=True)
        zip_name = os.path.join("./models", generate_zip_name("model"))
        zip_model("./whisper_malagasy_final", zip_name)
        print(f"Modèle sauvegardé dans : {zip_name}")
    else:
        os.makedirs("./models_inutilisables", exist_ok=True)
        zip_name = os.path.join("./models_inutilisables", generate_zip_name("unused_model"))
        zip_model("./whisper_malagasy_final", zip_name)
        print(f"Modèle inutilisable sauvegardé dans : {zip_name}")