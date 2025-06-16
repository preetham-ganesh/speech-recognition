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


def load_dataset_file_paths(split_name: str) -> List[Dict[str, str]]:
    """Loads file paths and transcription texts for a given dataset split (train, validation & test).

    Args:
        split_name: A string for the name of the current dataset split.

    Returns:
        A list of dictionaries for the dataset info in current data split.
    """
    # Asserts type & value of the arguments.
    assert isinstance(split_name, str) and split_name in [
        "train-360",
        "train-100",
        "validation",
        "test",
    ], "Variable split_name should be of type 'str' and have value as 'train-360', 'train-100', 'validation' or 'test'."

    # A dictionary to store the sub directory name based on the dataset split name.
    sub_directory_names = {
        "train-360": "train-clean-360",
        "train-100": "train-clean-100",
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


def load_preprocess_audio(file_path: str) -> np.ndarray:
    """Loads and preprocesses an audio file into a Short-time Fourier Transform.

    Args:
        file_path: A string for the absolute path of the file location.

    Returns:
        A NumPy array for the Short-time Fourier Transform loaded from the audio file.
    """
    # Asserts type & value of the arguments.
    assert isinstance(file_path, str), "Variable file_path should be of type 'str'."

    # Loads the audio for file path with sampling rate 16k and mono as default.
    audio, _ = librosa.load(file_path, sr=16000, mono=True)

    # Sets the STFT parameters.
    frame_length, frame_step, fft_length = 200, 80, 256

    # Computes STFT using scipy. 'hop_length' corresponds to frame_step, n_fft to fft_length.
    stfts = librosa.stft(
        audio,
        n_fft=fft_length,
        hop_length=frame_step,
        win_length=frame_length,
        window="hann",
    )

    # Takes magnitude and apply power of 0.5.
    x = np.abs(stfts) ** 0.5

    # Transposes to shape (time, frequency).
    x = x.T

    # Normalizes the STFT to subtract mean, divide by standard deviation along frequency axis. Avoids, division by 0.
    x_mean = np.mean(x, axis=1, keepdims=True)
    x_std_dev = np.std(x, axis=1, keepdims=True)
    x_std_dev = np.where(x_std_dev == 0, 1, x_std_dev)
    x = (x - x_mean) / x_std_dev

    # Replaces any NaN values with 0.
    x = np.where(np.isnan(x), 0.0, x)
    return x


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
    dataset_size: str,
) -> None:
    """Preprocesses audio files & their transcriptions in the current data split.

    Args:
        dataset_version: A string for the version of the processed dataset.
        split_name: A string for the name of the current dataset split.
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
    assert isinstance(dataset_size, str) and dataset_size in [
        "mini",
        "full",
    ], "Variable dataset_size should be of type 'str' and have value as 'mini' or 'full'."

    # Loads file paths and transcription texts for a given dataset split (train, validation & test).
    if split_name == "train":
        original_dataset_info = load_dataset_file_paths("train-100")

        # If dataset size is full, then loads the train-clean-360 file paths.
        if dataset_size == "full":
            original_dataset_info += load_dataset_file_paths("train-360")

    else:
        original_dataset_info = load_dataset_file_paths(split_name)
    print()

    # Checks if the following directory path exists.
    processed_data_directory_path = check_directory_path_existence(
        os.path.join(
            "data", "processed_data", "librispeech", f"v{dataset_version}", split_name
        )
    )

    # Iterates across file paths & transcription texts in original dataset.
    processed_dataset_info = list()
    n_examples = len(original_dataset_info)
    for r_id, row in enumerate(original_dataset_info):

        # Loads and preprocesses an audio file into a Short-time Fourier Transform.
        audio_stft = load_preprocess_audio(row["file_path"])

        # Preprocesses text string by stripping whitespace and converting to lowercase.
        processed_text = preprocess_text(row["text"])

        # Saves the log mel spectrogram as NumPy array
        file_path = os.path.join(processed_data_directory_path, f"{r_id}.npy")
        with open(file_path, "wb") as f:
            np.save(f, audio_stft)
            f.close()

        # Appends the processed file info as dictionary to the list.
        processed_dataset_info.append(
            {
                "file_path": file_path,
                "text": processed_text,
                "n_time_frames": audio_stft.shape[0],
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
    args = parser.parse_args()

    # Preprocesses audio files & their transcriptions in the current data split.
    preprocess_dataset(args.dataset_version, "train", args.dataset_size)
    preprocess_dataset(args.dataset_version, "validation", args.dataset_size)
    preprocess_dataset(args.dataset_version, "test", args.dataset_size)


if __name__ == "__main__":
    main()
