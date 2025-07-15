import os
from utils.gotique import make_gotique

base_dir = "datasets"
audios_dir = os.path.join(base_dir, "audios")
transcriptions_dir = os.path.join(base_dir, "transcriptions")

def construction():
    print(f"\n{make_gotique('Transcription ...')}")
    for folder_name in os.listdir(audios_dir):
        folder_path = os.path.join(audios_dir, folder_name)
        if os.path.isdir(folder_path):
            text_content = folder_name.replace('_', ' ')
            txt_file_path = os.path.join(transcriptions_dir, f"{folder_name}.txt")
            
            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read().strip()
                if existing_content == text_content:
                    continue
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)