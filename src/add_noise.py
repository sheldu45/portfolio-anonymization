import argparse
import os
import numpy as np
import soundfile as sf

class AudioWhiteNoiseAdder:
    """
    AudioWhiteNoiseAdder is a class that replaces specific segments of an audio file with white noise.
    
    Attributes:
        noise_level (float): The standard deviation of the white noise to be added, controlling its intensity.
    
    Methods:
        __init__(noise_level=0.02):
            Initializes the AudioWhiteNoiseAdder with the specified noise level.
        
        replace_with_white_noise(audio_path, timestamps, output_folder):
            Replaces specified time segments of the audio file with white noise and saves the result.
    """

    def __init__(self, noise_level=0.02):
        """
        Initializes the AudioWhiteNoiseAdder with the specified noise level.
        
        Args:
            noise_level (float): The intensity of the white noise. Default is 0.02.
        """
        self.noise_level = noise_level

    def replace_with_white_noise(self, audio_path, timestamps, output_folder):
        """
        Replaces specified time segments of the audio file with white noise.
        
        Args:
            audio_path (str): Path to the input audio file.
            timestamps (list of tuples): List of tuples specifying the start and end times (in seconds) for white noise replacement.
            output_folder (str): Folder path to save the output audio file with replaced white noise.
        
        This method reads the audio file, generates white noise for the specified segments, and replaces the original audio with the noise.
        """
        # Load the audio file
        data, samplerate = sf.read(audio_path)
        
        # Replace specified segments with white noise
        for start, end in timestamps:
            start_sample = int(start * samplerate)
            end_sample = int(end * samplerate)
            
            # Generate white noise and replace the audio data
            noise = np.random.normal(0, self.noise_level, end_sample - start_sample)
            data[start_sample:end_sample] = noise
        
        # Generate the output file name
        base_name = os.path.basename(audio_path)
        output_file_name = f"modified_{base_name}"
        output_path = os.path.join(output_folder, output_file_name)
        
        # Save the modified audio to the output path
        sf.write(output_path, data, samplerate)

if __name__ == "__main__":
    # Set up argument parsing with explicit naming
    parser = argparse.ArgumentParser(description="Replace specific sections of an audio file with white noise.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder path to save the output audio file.")
    parser.add_argument("--timestamps", type=str, help="List of tuples with start and end times in seconds. Example: '[(1.0, 2.0), (5.0, 7.0)]'", default="[]")
    parser.add_argument("--noise_level", type=float, help="Level of white noise to add. Default is 0.02.", default=0.02)
    
    args = parser.parse_args()
    
    # Convert the string of timestamps to a list of tuples
    timestamps = eval(args.timestamps)
    
    # Ensure the output directory exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Create an instance of AudioWhiteNoiseAdder and replace audio segments with white noise
    adder = AudioWhiteNoiseAdder(noise_level=args.noise_level)
    adder.replace_with_white_noise(args.audio_path, timestamps, args.output_folder)
    
    print(f"White noise replaced segments in {args.audio_path} and saved to {os.path.join(args.output_folder, 'modified_' + os.path.basename(args.audio_path))}")
