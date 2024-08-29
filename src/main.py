import os
from data_handler import DataHandler
from transcriber import Transcriber
from pii_detector import PIIDetector
from anonymizer import Anonymizer
import argparse
from pathlib import Path
import json
from dotenv import load_dotenv

class DataProcessor:
    """
    A class to handle the processing of an audio dataset, including transcription,
    PII detection, and anonymization. The dataset can be processed from a specified
    directory, and the results are saved in designated output directories.

    Attributes:
    - data_handler (DataHandler): Instance to handle data loading and management.
    - transcriber (Transcriber): Instance to handle audio transcription.
    - pii_detector (PIIDetector): Instance to detect PII in transcriptions.
    - anonymizer (Anonymizer): Instance to anonymize transcriptions and audio files.

    Methods:
    - process_data(path_to_raw_audio_files): Process the audio files, transcribe, detect PII, and anonymize.
    """

    def __init__(self, api_key, test_size, seed=42, reimport=False):
        """
        Initialize the DataProcessor class.

        Parameters:
        - api_key: str, API key for accessing transcription and PII detection services.
        - test_size: int, number of items to keep in the test set.
        - seed: int, random seed for reproducibility.
        - reimport: bool, whether to re-import the dataset even if it exists locally.
        """
        self.data_handler = DataHandler(test_size=test_size, seed=seed, reimport=reimport)
        self.transcriber = Transcriber(api_key)
        self.pii_detector = PIIDetector(api_key)
        self.anonymizer = Anonymizer(api_key)

    def process_data(self, path_to_raw_audio_files='data/raw_audios'):
        """
        Process the audio files in the specified directory. This includes transcribing
        the audio, detecting PII in the transcriptions, and anonymizing both the transcriptions
        and the audio files.

        Parameters:
        - path_to_raw_audio_files: str, path to the directory containing raw audio files.
        """
        # Resolve the path to raw audio files
        root_dir = Path(__file__).resolve().parent.parent
        path_to_raw_audio_files = root_dir / path_to_raw_audio_files
        
        # Loop through the audios
        for audio_path in self.data_handler:
            # Step 1 : Transcribe the audio
            transcription = self.transcriber.transcribe(audio_path, output_dir='data/transcriptions')

            # Step 2: Detect PII in the transcription
            pii_indexes = self.pii_detector.detect_pii_tokens(transcription)

            # Step 3: Anonymize the transcription and save it
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            anonymized_transcription = self.anonymizer.text_anonymization(transcription, pii_indexes, output_folder="data/anonymized_transcriptions", base_name=base_name)
            
        # Loop through the audios in data/raw_audios (because it means they are part of the test set)
        for audio_path in os.listdir(path_to_raw_audio_files):
            audio_path_full = path_to_raw_audio_files / audio_path

            # Step 0: Get the transcription file path
            transcription_file = root_dir / f"data/transcriptions/transcription_{audio_path_full.stem}.json"

            # Step 1: Read the transcription file into memory
            with open(transcription_file, 'r') as file:
                transcription = json.load(file)
            
            # Step 2: Detect PII in the audio
            pii_indexes_audio = self.pii_detector.detect_pii_tokens(transcription)

            # Step 3: Anonymize the audio and save it
            self.anonymizer.audio_anonymization(transcription, pii_indexes_audio, audio_path_full, output_folder="data/anonymized_audios")
            
        # Print a message to indicate the processing is complete
        print("Data processing complete.")

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file

    parser = argparse.ArgumentParser(description="Process and anonymize audio dataset.")
    parser.add_argument('--test_size', '-t', type=int, default=5, help='Number of items to keep in the test set.')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--reimport', '-r', action='store_true', help='Re-import the dataset even if it exists locally.')
    
    args = parser.parse_args()

    api_key = os.getenv('API_KEY')  # Retrieve the API key from environment variables
    
    if not api_key:
        raise ValueError("API_KEY not found in environment variables. Please set it in the .env file.")

    processor = DataProcessor(
        api_key=api_key,
        test_size=args.test_size,
        seed=args.seed,
        reimport=args.reimport
    )

    processor.process_data()