# Portfolio Audio Anonymizer

## Introduction

The **Portfolio Audio Anonymizer** is a Python-based tool designed for the anonymization of audio datasets. It leverages state-of-the-art technologies, including OpenAI's API, to transcribe audio files, detect Personally Identifiable Information (PII), and anonymize both the transcriptions and the audio itself. This project is especially useful for handling sensitive audio data that needs to be processed and shared without exposing personal information.

The core functionalities of this project include:
- **Transcription**: Converts audio files into text using OpenAI's ASR (Automatic Speech Recognition) capabilities.
- **PII Detection**: Identifies and indexes personal information in the transcriptions.
- **Anonymization**: Replaces identified PII in text with placeholders and obscures corresponding audio segments with white noise.

## Project Structure

The project is organized as follows:

```
portfolio-audio-anonymizer/
│
├── data/
│   ├── anonymized_audios/            # Stores anonymized audio files
│   ├── anonymized_transcriptions/    # Stores anonymized transcriptions
│   ├── dataset_local/                # Stores local copies of datasets
│   ├── raw_audios/                   # Stores original raw audio files
│   └── transcriptions/               # Stores transcriptions of the raw audio files
│
├── data_save/
│   ├── anonymized_audios/            # Sample of anonymized audio files from a prior instance
│   ├── anonymized_transcriptions/    # Sample of anonymized transcriptions from a prior instance
│   ├── dataset_local/                # Sample of datasets from a prior instance
│   ├── raw_audios/                   # Sample of raw audio files from a prior instance
│   └── transcriptions/               # Sample of transcriptions from a prior instance
│
├── src/
│   ├── data_handler.py               # Handles data loading and management, including reservoir sampling
│   ├── transcriber.py                # Manages audio transcription using OpenAI API
│   ├── pii_detector.py               # Detects PII in transcriptions
│   ├── anonymizer.py                 # Anonymizes text and audio based on PII detection
│   └── main.py                       # Entry point to run the full anonymization process
│
├── .env                              # Environment file for storing API keys
├── requirements.txt                  # List of dependencies for the project
├── README.md                         # Project documentation
└── .gitignore                        # Files and directories to ignore in version control
```

### Key Components

- **`data/`**: Directory containing raw, transcribed, and anonymized audio data.
- **`data_save/`**: Contains a sample of artifacts from a prior instance of code execution, mirroring the structure of the `data/` directory. This folder is useful for reference or comparison with current data processing runs.
- **`src/data_handler.py`**: Handles the loading and processing of audio datasets, including test set selection via reservoir sampling.
- **`src/transcriber.py`**: Facilitates transcription of audio files using the OpenAI API.
- **`src/pii_detector.py`**: Detects and indexes PII in the transcriptions.
- **`src/anonymizer.py`**: Manages the anonymization of detected PII in both text and audio formats.
- **`src/main.py`**: The main entry point that ties together the entire workflow—transcription, PII detection, and anonymization.

## How to Run

Follow these instructions to set up and run the project on your local machine.

### 1. Set Up a Virtual Environment

It's recommended to run the project in a virtual environment to manage dependencies cleanly.

```bash
# Create a virtual environment in a directory named `venv`
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

Once the virtual environment is active, install the required Python packages.

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

The project uses OpenAI's API for transcription and PII detection, which requires an API key. You need to create a `.env` file in the project's root directory to store this key.

Create a `.env` file with the following content:

```
API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

### 4. Run the Main Code

You can now run the project to process and anonymize audio files. Use the following command:

```bash
python src/main.py --test_size 5 --seed 42 --reimport
```

This will:

- Process the raw audio files in the `data/raw_audios` directory.
- Transcribe the audio files using OpenAI's ASR model.
- Detect and index PII in the transcriptions.
- Anonymize both the text and audio, saving the outputs in `data/anonymized_transcriptions` and `data/anonymized_audios`, respectively.

### Command-Line Arguments

- `--test_size` or `-t`: Number of items to keep in the test set (default: `5`).
- `--seed` or `-s`: Random seed for reproducibility (default: `42`).
- `--reimport` or `-r`: A flag to force re-importing the dataset even if it exists locally.

## Conclusion

This project offers a comprehensive solution for anonymizing sensitive audio data, making it suitable for various use cases, including research, data sharing, and compliance with privacy regulations. The `data_save` folder contains a sample of artifacts from a prior instance of code execution, which can be useful for reference or comparison with the current data processing runs.

Feel free to customize and extend the functionality based on your specific needs. For any issues or contributions, please refer to the project's repository.