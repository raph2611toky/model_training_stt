from .bruit_reduce import process_all_audio_files
from .construct_transcription import construction

if __name__ == "__main__":
    process_all_audio_files()
    construction()