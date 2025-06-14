import os
import sys
import warnings
import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")


import librosa
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

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
    dataset_info = list()
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
                dataset_info.append(
                    {
                        "file_path": os.path.join(
                            extracted_data_directory_path,
                            dir_0,
                            dir_1,
                            f"{file_name}.flac",
                        ),
                        "text": text,
                    }
                )
    print(
        f"Original no. of examples in the {split_name} data split: {len(dataset_info)}"
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

    # Adds audio normalization.
    y = librosa.util.normalize(y)

    # Computes log-mel spectrogram for the loaded audio file.
    spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=512, win_length=1280
    )
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


def preprocess_dataset(
    dataset_version: str,
    split_name: str,
    n_mels: int,
    dataset_size: str,
) -> None:
    """Preprocesses audio files & their transcriptions in the current data split.

    Preprocesses audio files & their transcriptions in the current data split.

    Args:
        dataset_version: A string for the version of the processed dataset.
        split_name: A string for the name of the current dataset split.
        n_mels: An integer for the no. of frequency bins to be computed.
        dataset_size: A string for the size of the processed dataset.

    Returns:
        None.
    """
    # Asserts type & value of the arguments.
    assert isinstance(
        dataset_version, str
    ), "Variable dataset_version should be of type 'str'."
    assert isinstance(split_name, str) and split_name in [
        "train",
        "validation",
        "test",
    ], "Variable split_name should be of type 'str' and have value as 'train', 'validation' or 'test'."
    assert isinstance(n_mels, int), "Variable n_mels should be of type 'int'."
    assert isinstance(dataset_size, str) and dataset_size in [
        "mini",
        "full",
    ], "Variable dataset_size should be of type 'str' and have value as 'mini' or 'full'."

    # Loads file paths and transcription texts for a given dataset split (train, validation & test).
    original_dataset_info = load_dataset_file_paths(split_name)
    print()

    # Checks if the following directory path exists.
    processed_data_directory_path = check_directory_path_existence(
        os.path.join(
            "data", "processed_data", "librispeech", f"v{dataset_version}", split_name
        )
    )

    # If split is 'train' and dataset size is 'mini', then only 30% of processed dataset is processed.
    n_examples = len(original_dataset_info)
    if split_name == "train" and dataset_size == "mini":
        n_examples = int(n_examples * 0.3)

    # Iterates across file paths & transcription texts in original dataset.
    processed_dataset_info = list()
    for r_id, row in enumerate(original_dataset_info[:n_examples]):

        # Loads and preprocesses an audio file into a log-Mel spectrogram.
        log_spectrogram = load_preprocess_audio(row["file_path"], n_mels)

        # Preprocesses text string by stripping whitespace and converting to lowercase.
        processed_text = preprocess_text(row["text"])

        # Saves the log mel spectrogram as NumPy array
        file_path = os.path.join(processed_data_directory_path, f"{r_id}.npy")
        with open(file_path, "wb") as f:
            np.save(f, log_spectrogram)
            f.close()

        # Appends the processed file info as dictionary to the list.
        processed_dataset_info.append(
            {
                "file_path": file_path,
                "text": processed_text,
                "n_time_frames": log_spectrogram.shape[0],
            }
        )

        if r_id != 0 and r_id % 1000 == 0:
            print(
                f"Finished processing {((r_id / n_examples) * 100):.3f}% in the {split_name} data split."
            )
    print()

    # Shuffles the processed dataset info.
    processed_dataset_info = shuffle(processed_dataset_info, random_state=42)

    # Converts list of dictionaries into pandas dataframe.
    processed_dataset_info = pd.DataFrame.from_records(processed_dataset_info)
    print(
        f"No. of examples in the {split_name} data split: {len(processed_dataset_info)}"
    )
    print(
        f"Maximum time frames in the {split_name} data split: {max(processed_dataset_info['n_time_frames'])}"
    )
    print()

    # Saves processed dataset info as a CSV file.
    processed_dataset_info.to_csv(
        os.path.join(processed_data_directory_path, "dataset_info.csv")
    )


def main():
    print()

    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dv",
        "--dataset_version",
        type=str,
        required=True,
        help="Enter the version by which the processed dataset should be saved as.",
    )
    parser.add_argument(
        "-ds",
        "--dataset_size",
        type=str,
        required=True,
        help="Enter the size of the processed dataset.",
    )
    parser.add_argument(
        "-nm",
        "--n_mels",
        type=int,
        required=True,
        help="Enter the no. of frequency bins that should be computed.",
    )
    args = parser.parse_args()

    # Preprocesses audio files & their transcriptions in the current data split.
    preprocess_dataset(
        args.dataset_version,
        "train",
        args.n_mels,
        args.dataset_size,
    )
    preprocess_dataset(
        args.dataset_version, "validation", args.n_mels, args.dataset_size
    )
    preprocess_dataset(args.dataset_version, "test", args.n_mels, args.dataset_size)


if __name__ == "__main__":
    main()
