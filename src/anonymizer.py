import json
import os
import argparse
from pathlib import Path
from add_noise import AudioWhiteNoiseAdder
from openai import OpenAI

class Anonymizer:
    """
    Anonymizer class for text and audio anonymization tasks.
    This class provides methods for anonymizing text and audio data. It replaces specific tokens in text data with "<PII>"
    and replaces specific timestamps in audio data with white noise.
    Attributes:
        audio_white_noise_adder (AudioWhiteNoiseAdder): An instance of the AudioWhiteNoiseAdder class.
        client (OpenAI): An instance of the OpenAI class for making API calls.
    Methods:
        __init__(api_key): Initializes the Anonymizer class, setting up the AudioWhiteNoiseAdder instance and OpenAI client.
        text_anonymization(json_data, token_indexes, language='french', output_folder=None, base_name=None): Anonymizes the text data by replacing specific tokens with "<PII>".
        audio_anonymization(json_data, token_indexes, audio_path, output_folder=None): Anonymizes the audio data by replacing specific timestamps with white noise.
    """
    
    def __init__(self, api_key):
        """
        Initializes the Anonymizer class, setting up the AudioWhiteNoiseAdder instance and OpenAI client.
        
        This constructor prepares the Anonymizer for both text and audio anonymization tasks.
        
        Args:
            api_key (str): The API key for accessing the OpenAI API.
        """
        self.audio_white_noise_adder = AudioWhiteNoiseAdder()
        self.client = OpenAI(api_key=api_key)

    def text_anonymization(self, json_data, token_indexes, language='french', output_folder=None, base_name=None):
        """
        Anonymizes the text data by replacing specific tokens with "<PII>".
        
        Args:
            json_data (str or dict): Either the JSON data directly or the path to the JSON file containing the text data.
            token_indexes (list of int): List of token indexes to be anonymized.
            language (str, optional): Language of the text data. Defaults to 'french'.
            output_folder (str, optional): Folder path where the anonymized text file will be saved.
            base_name (str, optional): Base name of the output file. If not provided, a default name will be used.
        
        This method reads the JSON data, identifies the tokens that need to be anonymized based on the provided indexes, 
        and replaces those tokens with "<PII>". The resulting anonymized text is returned or saved to a file if an output folder is specified.
        """
        # Resolve the absolute path of the output folder
        if output_folder:
            output_folder = Path(__file__).resolve().parent.parent / output_folder
        
        # Load the JSON data if json_data is a file path
        if isinstance(json_data, str):
            with open(json_data, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
        # Anonymize the text data
        words = []
        for i, token in enumerate(json_data):
            if i in token_indexes:
                if not words or words[-1] != "<PII>":
                    words.append("<PII>")
            else:
                words.append(token['word'])

        result = ' '.join(words)
        
        # Diacritise the text
        prompt_diacritization = (
            "The following text is a direct transcript of a phone conversation between an emergency medical services operator and a caller."
            "The conversation is presented as a continuous dialogue without identifying the speakers by name or role, which can make it challenging to follow who is speaking at any given time."
            "The conversation also lacks punctuation, which can make it difficult to determine where sentences begin and end."
            "Your task is to reformat conversation into a more readable format by adding punctuation and speaker labels."
            "Include to your answer no introduction, nor conclusion, no title, just the conversation."
            "Conversation is in {language}."
            "Input text: {input_text}"
        ).format(input_text=result, language=language)

        # Get the completion from OpenAI API
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt_diacritization,
                }
            ],
            model="gpt-4",
        )        

        diarized_result = response.choices[0].message.content        

        if output_folder:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                
            if base_name is None:
                base_name = os.path.basename(json_data) if isinstance(json_data, str) else "anonymized.json"
                
            output_file_name = f"anonymized_{base_name}.txt"
            output_path = os.path.join(Path(__file__).resolve().parent.parent, output_folder, output_file_name)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(diarized_result)
        else:
            return diarized_result

    def audio_anonymization(self, json_data, token_indexes, audio_path, output_folder=None):
        """
        Anonymizes the audio data by replacing specific timestamps with white noise.
        
        Args:
            json_data (str or dict): Either the JSON data directly or the path to the JSON file containing the audio data.
            token_indexes (list of int): List of token indexes to be anonymized.
            audio_path (str): Path to the audio file to be anonymized.
            output_folder (str, optional): Folder path where the anonymized audio file will be saved.
        
        This method reads the JSON data, identifies the timestamps that need to be anonymized based on the provided indexes, 
        and replaces those timestamps with white noise. The resulting anonymized audio is saved to a file if an output folder is specified.
        """
        # Load the JSON data if json_data is a file path
        if isinstance(json_data, str):
            with open(json_data, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
        # Anonymize the audio data
        timestamps = []
        current_start = None

        for i in token_indexes:
            token = json_data[i]
            if current_start is None:
                current_start = token['start']
            current_end = token['end']
            
            if i == token_indexes[-1] or token_indexes[token_indexes.index(i) + 1] != i + 1:
                timestamps.append((current_start, current_end))
                current_start = None

        # Call the function from add_noise.py to replace with white noise
        self.audio_white_noise_adder.replace_with_white_noise(audio_path, timestamps, output_folder)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Anonymize text and audio data.")
    parser.add_argument('--json_file', help='Path to the JSON file containing the text data.')
    parser.add_argument('--indexes', nargs='+', type=int, help='List of token indexes to anonymize.')
    parser.add_argument('--audio_file', help='Path to the audio file to anonymize.')
    parser.add_argument('--output_folder', help='Path to the output folder where results should be saved.')
    parser.add_argument('-k', '--api_key', required=True, help="OpenAI API key")
    args = parser.parse_args()

    anonymizer = Anonymizer(args.api_key)

    if args.json_file and args.indexes:
        
        with open(args.json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        anonymizer.text_anonymization(json_data, args.indexes, args.output_folder)
        
        if args.audio_file:
            anonymizer.audio_anonymization(json_data, args.indexes, args.audio_file, args.output_folder)
