import json
import os
from add_noise import replace_with_white_noise  # Import the function from add_noise.py

class Anonymizer:

    def __init__(self):
        pass

    def text_anonymization(self, json_data, token_indexes, output_folder=None):
        words = []
        for i, token in enumerate(json_data):
            if i in token_indexes:
                if not words or words[-1] != "<PII>":
                    words.append("<PII>")
            else:
                words.append(token['word'])

        result = ' '.join(words)

        if output_folder:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path = os.path.join(output_folder, 'anonymized_text.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        else:
            return result

    def audio_anonymization(self, json_data, token_indexes, audio_path, output_folder=None):
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
        replace_with_white_noise(audio_path, timestamps, output_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anonymize text and audio data.")
    parser.add_argument('--json_file', help='Path to the JSON file containing the text data.')
    parser.add_argument('--indexes', nargs='+', type=int, help='List of token indexes to anonymize.')
    parser.add_argument('--audio_file', help='Path to the audio file to anonymize.')
    parser.add_argument('--output_folder', help='Path to the output folder where results should be saved.')
    args = parser.parse_args()

    anonymizer = Anonymizer()

    if args.json_file and args.indexes:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        anonymizer.text_anonymization(json_data, args.indexes, args.output_folder)
        
        if args.audio_file:
            anonymizer.audio_anonymization(json_data, args.indexes, args.audio_file, args.output_folder)
