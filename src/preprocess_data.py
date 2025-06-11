import os
import sys
import warnings
import time


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")


import librosa
import numpy as np

from src.utils import check_directory_path_existence, load_text_file

from typing import Dict, List


def load_dataset_file_paths(split_name: str) -> Dict[str, List[str]]:
    """Loads file paths and transcription texts for a given dataset split (train, validation & test).

    Loads file paths and transcription texts for a given dataset split (train, validation & test).

    Args:
        split_name: A string for the name of the current dataset split.

    Returns:
        A dictionary for the dataset info in current data split.
    """
    # Asserts type & value of the arguments.
    assert isinstance(split_name, str) and split_name in [
        "train",
        "validation",
        "test",
    ], "Variable split_name should be of type 'str' and have value as 'train', 'validation' or 'test'."

    # A dictionary to store the sub directory name based on the dataset split name.
    sub_directory_names = {
        "train": "train-clean-360",
        "validation": "dev-clean",
        "test": "test-clean",
    }

    # Checks if the following directory path exists.
    extracted_data_directory_path = check_directory_path_existence(
        os.path.join(
            "data",
            "extracted_data",
            "librispeech",
            split_name,
            "LibriSpeech",
            sub_directory_names[split_name],
        )
    )

    # Iterates across directories in the extracted data directory.
    dataset_info = {"file_path": list(), "text": list()}
    for dir_0 in os.listdir(extracted_data_directory_path):

        # Iterates across directory in current directory.
        for dir_1 in os.listdir(os.path.join(extracted_data_directory_path, dir_0)):

            # Loads text file as a string.
            transcriptions = load_text_file(
                f"{dir_0}-{dir_1}.trans.txt",
                os.path.join(extracted_data_directory_path, dir_0, dir_1),
            ).split("\n")

            # Iterates across transcriptions in current directory.
            for file_info in transcriptions:

                # Splits file info into file name & transcription text.
                try:
                    file_name, text = file_info.split(" ", 1)
                except ValueError:
                    continue

                # Appends absolute file path & transcription text for current file into dataset info.
                dataset_info[split_name]["file_path"].append(
                    os.path.join(
                        extracted_data_directory_path,
                        dir_0,
                        dir_1,
                        f"{file_name}.flac",
                    )
                )
                dataset_info[split_name]["text"].append(text)

    print(
        f"No. of examples in the {split_name} data split: {len(dataset_info['file_path'])}"
    )
    return dataset_info


def load_preprocess_audio(file_path: str, n_mels: int) -> np.ndarray:
    """Loads and preprocesses an audio file into a log-Mel spectrogram.

    Loads and preprocesses an audio file into a log-Mel spectrogram.

    Args:
        file_path: A string for the absolute path of the file location.
        n_mels: An integer for the no. of frequency bins to be computed.

    Returns:
        A NumPy array for the log-mel spectrogram loaded from the audio file.
    """
    # Asserts type & value of the arguments.
    assert isinstance(file_path, str), "Variable file_path should be of type 'str'."
    assert isinstance(n_mels, int), "Variable n_mels should be of type 'int'."

    # Loads audio using the file path, with sample rate at 16kHz.
    y, sr = librosa.load(file_path, sr=16000)

    # Computes log-mel spectrogram for the loaded audio file.
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Transposes: librosa spectrogram from (freq_bins, time_steps) -> (time_steps, freq_bins).
    log_spectrogram = log_spectrogram.T
    return log_spectrogram


def preprocess_text(text: str) -> str:
    """Preprocesses text string by stripping whitespace and converting to lowercase.

    Preprocesses text string by stripping whitespace and converting to lowercase.

    Args:
        text: A string for the transcription text in current file that should be processed.

    Returns:
        A processed version of the transcription text.
    """
    # Strip leading & trailing whitespace.
    text = text.strip()

    # Converts all characters in text to lowercase.
    text = text.lower()
    return text
