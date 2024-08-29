import os
import time
import random
import soundfile as sf
from datasets import load_dataset, load_from_disk
import argparse


class DataHandler:
    """
    A class to handle the processing of an audio dataset with reservoir sampling
    to select a test set of a specified size. The dataset can be loaded from 
    Hugging Face or from a local disk, and audio files are saved and managed 
    within a specified output directory.

    Attributes:
    - dataset_name (str): Name of the dataset to load from Hugging Face.
    - output_dir (str): Directory where the decoded audio files will be saved.
    - test_size (int): Number of items to keep in the test set.
    - seed (int): Random seed for reproducibility.
    - reimport (bool): Whether to re-import the dataset even if it exists locally.

    Methods:
    - __iter__(): Initialize the iterator for the dataset.
    - __next__(): Process the next item in the dataset, applying reservoir sampling.
    - delete_previous_audio(): Delete the previously saved audio file.
    - _apply_reservoir_sampling(idx): Apply reservoir sampling to decide whether 
      to keep the previous audio in the test set.
    - _delete_audio_by_index(idx): Delete the audio file by its index.
    - _finalize_test_set(): Ensure the final element is considered for the test set.
    - _print_test_set_summary(): Print a summary of the test set at the end of the process.
    """

    def __init__(self, dataset_name='diarizers-community/simsamu', output_dir=None, test_size=5, seed=42, reimport=False):
        """
        Initialize the DataHandler class.

        Parameters:
        - dataset_name: str, name of the dataset to load from Hugging Face.
        - output_dir: str, directory where the decoded audio files will be saved.
        - test_size: int, number of items to keep in the test set.
        - seed: int, random seed for reproducibility.
        - reimport: bool, whether to re-import the dataset even if it exists locally.
        """
        # Determine the default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '../data/raw_audios')

        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.test_size = test_size
        self.seed = seed

        # Set random seed
        random.seed(self.seed)

        # Define a path for saving the dataset locally, outside the raw_audios folder
        self.local_path = os.path.join(os.path.dirname(self.output_dir), 'dataset_local')

        # Load the dataset if it's missing or if reimport is True
        if not os.path.exists(self.local_path) or reimport:
            print("Loading dataset from Hugging Face and saving locally...")
            self.dataset = load_dataset(self.dataset_name, split='train')
            os.makedirs(self.local_path, exist_ok=True)
            self.dataset.save_to_disk(self.local_path)
        else:
            try:
                print("Loading dataset from local storage...")
                self.dataset = load_from_disk(self.local_path)
            except FileNotFoundError:
                print("Dataset not found locally. Re-importing dataset...")
                self.dataset = load_dataset(self.dataset_name, split='train')
                os.makedirs(self.local_path, exist_ok=True)
                self.dataset.save_to_disk(self.local_path)

        # Clear any existing raw audio files from previous runs
        self._clear_raw_audio()

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize the current index for iteration
        self.current_idx = 0
        self.num_kept = 0  # Track the number of audios kept for the test set

    def _clear_raw_audio(self):
        """
        Delete any raw audio files that remain from previous runs.
        """
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted old audio file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

    def __iter__(self):
        """
        Initialize the iterator.

        Returns:
        - self: DataHandler object ready for iteration.
        """
        self.current_idx = 0
        self.num_kept = 0
        return self

    def __next__(self):
        """
        Move to the next item in the dataset.

        Returns:
        - output_path: str, path to the saved audio file.

        Raises:
        - StopIteration: if the dataset has been fully iterated over.
        """
        if self.current_idx >= len(self.dataset):
            # Ensure the last element is considered for the test set before stopping
            self._finalize_test_set()
            self._print_test_set_summary()
            raise StopIteration

        # Access the current audio entry from the "train" split
        audio_entry = self.dataset[self.current_idx]['audio']

        # Decode the audio entry and save it as a .wav file
        output_path = os.path.join(self.output_dir,
                       f'audio_{self.current_idx}.wav')
        sf.write(output_path, audio_entry['array'],
             audio_entry['sampling_rate'])

        # Check the size of the saved audio file
        file_size = os.path.getsize(output_path)
        if file_size > 25 * 1024 * 1024:  # 25 MB in bytes
            os.remove(output_path)  # Remove the oversized audio file
            raise ValueError(f"Decoded audio file {output_path} is larger than 25 MB. Chunking is required.")
        
        # Apply reservoir sampling logic
        if self.current_idx < self.test_size:
            # Keep the first `test_size` elements
            self.num_kept += 1
            print(f"Audio {self.current_idx} kept for test set (Initial fill).")
        else:
            # Apply reservoir sampling after the first `test_size` elements
            self._apply_reservoir_sampling(self.current_idx)

        # Increment the index for the next iteration
        self.current_idx += 1

        # Return the path to the saved audio file
        return output_path

    def delete_previous_audio(self):
        """
        Delete the previously saved audio file.
        """
        if self.current_idx > 0:
            # Construct the path to the previous audio file
            audio_file = os.path.join(self.output_dir,
                                      f'audio_{self.current_idx - 1}.wav')

            # Remove the file if it exists
            if os.path.exists(audio_file):
                os.remove(audio_file)
                print(f"Deleted: {audio_file}")

    def _apply_reservoir_sampling(self, idx):
        """
        Apply reservoir sampling to decide whether to keep the previous audio in the test set.

        Parameters:
        - idx: int, index of the current audio file.
        """
        j = random.randint(0, idx)
        if j < self.test_size:
            # Replace the j-th element in the reservoir with the current element
            print(f"{j}-th audio in reservoir removed (Replaced by audio {idx}).")
            # Delete the previous file at index j
            self._delete_audio_by_index(j)
            # Increment the number of kept elements
            print(f"Audio {idx} kept for test set (Reservoir sampling logic).")
        else:
            self.delete_previous_audio()

    def _delete_audio_by_index(self, idx):
        """
        Delete the audio file by its index.

        Parameters:
        - idx: int, index of the audio file to delete.
        """
        audio_files = sorted([
            f for f in os.listdir(self.output_dir)
            if os.path.isfile(os.path.join(self.output_dir, f))
        ])
        if idx < len(audio_files):
            audio_file = os.path.join(self.output_dir, audio_files[idx])
            if os.path.exists(audio_file):
                os.remove(audio_file)
                print(f"Succesfully deleted audio {audio_files[idx]}.")
        else:
            print(f"Audio {idx} does not exist.")

    def _finalize_test_set(self):
        """
        Ensure the final element is considered for the test set.
        """
        if self.current_idx == len(self.dataset):
            self._apply_reservoir_sampling(self.current_idx - 1)

    def _print_test_set_summary(self):
        """
        Print a summary of the test set at the end of the process.
        """
        test_set_files = [
            f for f in os.listdir(self.output_dir)
            if os.path.isfile(os.path.join(self.output_dir, f))
        ]
        print(f"\nTest set summary:")
        test_set_files = sorted(test_set_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        print(f"Total files in test set: {len(test_set_files)}")
        for file in test_set_files:
            print(f"- {file}")


def main(dataset_name, output_dir, latency, test_size, seed, reimport):
    """
    Main function to handle the DataHandler class with user-defined arguments.

    Parameters:
    - dataset_name: str, name of the dataset to load from Hugging Face.
    - output_dir: str, directory where the decoded audio files will be saved.
    - latency: int, number of seconds to wait between processing each audio file.
    - test_size: int, number of items to keep in the test set.
    - seed: int, random seed for reproducibility.
    - reimport: bool, whether to re-import the dataset even if it exists locally.
    """
    # Instantiate the DataHandler with provided arguments
    handler = DataHandler(dataset_name, output_dir, test_size, seed, reimport)

    # Iterate through the dataset and process each audio entry
    for audio_path in handler:
        print(f"Processed: {audio_path}")
        
        # Wait for the specified latency before processing the next file
        time.sleep(latency)

        # Reservoir sampling is applied within the handler, no need for additional code here


if __name__ == "__main__":
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="Process and decode audio dataset entries."
    )

    # Determine the default output directory relative to the script location
    default_output_dir = os.path.join(os.path.dirname(__file__), '../data/raw_audios')

    # Add arguments for dataset name, output directory, latency, test size, seed, and reimport
    parser.add_argument(
        '-d', '--dataset', type=str,
        default='diarizers-community/simsamu',
        help="Name of the dataset to load (default: 'diarizers-community/simsamu')."
    )
    parser.add_argument(
        '-o', '--output', type=str,
        default=default_output_dir,
        help=f"Directory to save decoded audio files (default: '{default_output_dir}')."
    )
    parser.add_argument(
        '-l', '--latency', type=int,
        default=0,
        help="Number of seconds to wait between processing each audio file (default: 2)."
    )
    parser.add_argument(
        '-t', '--test_size', type=int,
        default=5,
        help="Number of entries to include in the test set (default: 5)."
    )
    parser.add_argument(
        '-s', '--seed', type=int,
        default=42,
        help="Random seed for reservoir sampling (default: 42)."
    )
    parser.add_argument(
        '-r', '--reimport', action='store_true',
        help="Re-import the dataset from Hugging Face even if it exists locally."
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.dataset, args.output, args.latency, args.test_size, args.seed, args.reimport)
