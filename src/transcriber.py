import json
import os
import argparse
from openai import OpenAI

class Transcriber:
    
    """
    A class to handle the transcription of audio files using OpenAI API.
    Attributes:
    - api_key (str): API key for accessing the OpenAI API.
    Methods:
    - __init__(api_key): Initializes the Transcriber object with the provided API key.
    - transcribe(file_path, output_dir=None): Transcribes the audio and returns the transcription as a JSON object. 
        If output_dir is specified, the transcription is also saved to a file in the output directory.
    """
    
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def transcribe(self, file_path, output_dir=None):
        # Ensure the output directory exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            ).words

        # If output_dir is specified, write transcription to a file
        if output_dir:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file_path = os.path.join(output_dir, f"transcription_{file_name}.json")
            with open(output_file_path, "w") as f:
                json.dump(transcription, f)

        return transcription

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI's Whisper model.")
    parser.add_argument(
        "-i", "--input",
        help="Path to the input audio file.",
        required=True
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="Directory to save the transcription file. Defaults to './data/transcriptions'.",
        default="./data/transcriptions"
    )
    parser.add_argument(
        "-k", "--api_key",
        help="Your OpenAI API key.",
        required=True
    )
    
    args = parser.parse_args()

    transcriber = Transcriber(api_key=args.api_key)
    transcription = transcriber.transcribe(file_path=args.input, output_dir=args.output_dir)

    print("Transcription completed.")
    print(transcription)

if __name__ == "__main__":
    main()
