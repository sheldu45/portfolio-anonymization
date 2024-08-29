import os
import json
import argparse
from openai import OpenAI
from collections import defaultdict

class PIIDetector:
    """
    PIIDetector is a class that uses OpenAI's API to detect Personal Identifiable Information (PII) in a given text.
    
    Attributes:
        client (OpenAI): An instance of the OpenAI client initialized with the provided API key.
    
    Methods:
        __init__(api_key):
            Initializes the PIIDetector with the given API key.
        
        _detect_pii_chunks(json_data, language="french"):
            Detects potential PII chunks in the provided JSON data using OpenAI's API.
        
        detect_pii_tokens(json_data, language="french"):
            Detects and returns the indexes of PII tokens in the provided JSON data.
    """
    
    def __init__(self, api_key):
        """
        Initializes the PIIDetector with the given API key.
        
        Args:
            api_key (str): The API key for accessing OpenAI's API.
        """
        self.client = OpenAI(api_key=api_key)

    def _detect_pii_chunks(self, json_data, language="french"):
        """
        Detects potential PII chunks in the provided JSON data using OpenAI's API.
        
        Args:
            json_data (list): A list of dictionaries containing tokenized text data.
            language (str): The language of the input text. Default is "french".
        
        Returns:
            list: A list of confirmed PII chunks.
        """
        # Extract the list of tokens from the JSON
        tokens = [entry['word'] for entry in json_data]

        # Join the tokens into a single string
        str_tokens = " ".join(tokens)

        # Create the prompt for the LLM initial detection using .format() with explicit variable naming
        prompt_inital_detection = (
            "In the following text, detect and list any chunks that is a potential Personal Identifiable Information (PII), "
            "Chunks are typically (but not necesarilly) composed of multiple tokens."
            "The language is {language}."
            "Include no introduction, nor conclusion, no title, just the list of PII chunks, comma separated."
            "Input text: {str_tokens}"
        ).format(str_tokens=str_tokens, language=language)

        # Get the completion from OpenAI API
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt_inital_detection,
                }
            ],
            model="gpt-4",
        )        

        initial_results = response.choices[0].message.content
        
        # Split the initial_results by comma (and strip whitespace, apostrophes, and quotes)
        pii_chunks = [chunk.strip(" '") for chunk in initial_results.split(",")]
                
        # Create the prompt for the LLM confirmation using
        prompt_confirm_pii = (
            "Is the following {language} chunk a Personal Identifiable Information (PII) ? Respond with 'yes' or 'no'."
            "examples of valid PII chunks:"
            "- names: John, Mary, Smith"
            "- surnames: Doe, Johnson, Brown"
            "- identifiable contact: email, phone number, address"
            "- identifiable personal information: social security number, passport number"
            "- specific symptoms: hematuria, paresthesia, diplopia, tachypnea, daltonism, photophobia, blood in urine, double vision, shortness of breath, sensitivity to light"
            "- specific or severe diseases: COVID-19, cancer, diabetes, heart attack, parkinson's disease, alzheimer's disease, epilepsy, sclerosis, crohn's disease, stroke, renal disease"
            "examples of invalid PII :"
            "- relative terms: mother, father, sister, daughter, son, cousin"
            "- titles: Mr., Mrs., Dr., Doctor, Professor"
            "- relative positions: boss, colleague, friend"
            "- relative locations: home, 3rd floor, office, school"
            "- generic dates: today, tomorrow, yesterday"
            "- generic locations: city, country, continent"
            "- generic possessions: his table, her car, his cat"
            "- generic symptoms: pain, discomfort, unease, tiredness, fatigue, age"
            "- generic or common diseases: flu, cold, headache, stomach ache"
            "- generic treatments: aspirin, paracetamol, ibuprofen"
            "- generic professions: doctor, teacher, engineer"
            "- company names: Google, Apple, Microsoft"
            "- product names: iPhone, Windows, Android"
            "- organization names: WHO, UNICEF, Red Cross"
            "The language is {language}."
            "Proper nouns may not be capitalized."
            "Chunk: {chunk}"
        )
        
        # Loop through each PII chunk and ask the LLM to confirm if it is a PII
        confirmed_pii_chunks = []  # Define the pii_chunks list
        for chunk in pii_chunks:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": prompt_confirm_pii.format(language=language, chunk=chunk),
                    }
                ],
                model="gpt-4",
            )
            confirmation = response.choices[0].message.content.lower()
            if confirmation.lower() == "yes":
                # Chunk is confirmed as PII, add it to the list of results
                confirmed_pii_chunks.append(chunk)

        return confirmed_pii_chunks
    
    def detect_pii_tokens(self, json_data, language="french"):
        """
        Detects and returns the indexes of PII tokens in the provided JSON data.
        
        Args:
            json_data (list): A list of dictionaries containing tokenized text data.
            language (str): The language of the input text. Default is "french".
        
        Returns:
            list: A list of indexes of PII tokens.
        """
        # Extract the list of tokens from the JSON and normalize (lowercase)
        list_tokens = [entry['word'].lower() for entry in json_data]
        
        # Detect PII chunks using the _detect_pii_chunks method
        pii_chunks = self._detect_pii_chunks(json_data, language="french")
                
        # For each chunk, split into tokens and normalize (lowercase)
        pii_chunks = [token.lower().split() for token in pii_chunks]
                
        # Initialize the lookup dictionary for PII chunks
        pii_chunks_dict = defaultdict(lambda: defaultdict(dict))

        # Loop through each PII chunk and build the lookup dictionary
        for tokens in pii_chunks:
            current_dict = pii_chunks_dict
            for i, token in enumerate(tokens):
                if i == len(tokens) - 1:
                    current_dict[token] = True  # Mark the end of the chunk
                else:
                    if token not in current_dict:
                        current_dict[token] = defaultdict(dict)
                        current_dict = current_dict[token]
                
        # Loop through token list and list indexes of PII tokens using pii_chunks_dict
        pii_token_indexes = []  # Initialize the list of PII token indexes

        # Loop through token list and list indexes of PII tokens using pii_chunks_dict
        for i in range(len(list_tokens)):
            current_dict = pii_chunks_dict
            for j in range(i, len(list_tokens)):
                token = list_tokens[j]
                if token in current_dict:
                    current_dict = current_dict[token]
                    if current_dict == True:
                        # End of PII chunk found, add tokens to pii_token_indexes
                        pii_token_indexes.extend(range(i, j + 1))
                        break
                else:
                    break

        return pii_token_indexes
    
def main():
    """
    Main function to run the PII Detector script.
    
    Parses command-line arguments, loads the input JSON data, and detects PII tokens.
    Prints the detected PII token indexes to stdout or saves them to an output file.
    """
    parser = argparse.ArgumentParser(description="PII Detector using OpenAI")
    parser.add_argument('-i', '--input', required=True, help="Path to the input JSON file")
    parser.add_argument('-o', '--output', required=False, help="Path to save the output (prints to stdout if not provided)")
    parser.add_argument('-l', '--language', required=False, default="french", help="Language of the input tokens")
    parser.add_argument('-k', '--api_key', required=True, help="OpenAI API key")

    args = parser.parse_args()

    # Load API key
    api_key = args.api_key

    # Initialize the PII Detector
    pii_detector = PIIDetector(api_key=api_key)

    # Load the input JSON data
    with open(args.input, 'r') as file:
        json_data = json.load(file)

    # Detect indexes of PII tokens
    pii_token_indexes = pii_detector.detect_pii_tokens(json_data=json_data, language=args.language)    

    # Print to stdout or save to the output file
    if args.output:
        with open(args.output, 'w') as output_file:
            output_file.write(pii_token_indexes)
        print(f"Detected indexes of PII token saved to {args.output}")
    else:
        print("Detected indexes of PII token:")
        print(pii_token_indexes)

if __name__ == "__main__":
    main()