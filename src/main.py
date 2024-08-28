from data_handler import DataHandler
from transcriber import Transcriber

def transcribe_audios(data_handler, transcriber):
    transcriptions = {}
    for idx, audio in enumerate(data_handler.train_set):
        audio_path = audio['path']  # Assuming 'path' contains the file path to the audio
        transcription = transcriber.transcribe(audio_path)
        transcriptions[audio_path] = transcription
        print(f"Transcribed {idx + 1}/{len(data_handler.train_set)}: {audio_path}")
    return transcriptions

if __name__ == "__main__":
    handler = DataHandler(dataset_name="diarizers-community/simsamu", test_size=20)
    handler.download_dataset()
    handler.split_dataset()
    
    transcriber = Transcriber(model_name="base")
    transcriptions = transcribe_audios(handler, transcriber)
    
    # Optional: Save transcriptions to a file for later use
    with open("transcriptions.json", "w") as f:
        json.dump(transcriptions, f, indent=2)
