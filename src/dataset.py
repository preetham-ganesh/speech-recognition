import os
import time
import tarfile

import requests
from tqdm import tqdm
import tensorflow as tf

from src.utils import check_directory_path_existence, load_text_file

from typing import Dict, Any, List


class Dataset(object):
    """"""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Creates object attributes for the Dataset class.

        Creates object attributes for the Dataset class.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initalizes class variables.
        self.model_configuration = model_configuration
        self.dataset_info = {
            "train": {"file_path": list(), "text": list()},
            "validation": {"file_path": list(), "text": list()},
            "test": {"file_path": list(), "text": list()},
        }

    def download_dataset(self) -> None:
        """Downloads the LibriSpeech dataset using the OpenSLR links.

        Downloads the LibriSpeech dataset using the OpenSLR links.

        Args:
            None.

        Returns:
            None.
        """
        # A dictionary for the data split based dataset links.
        dataset_links = {
            "test": "https://www.openslr.org/resources/12/test-clean.tar.gz",
            "validation": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
            "train": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        }

        # Checks if the following directory path exists.
        dataset_directory_path = check_directory_path_existence(
            os.path.join("data", "raw_data", "librispeech")
        )

        # Iterates across dataset links.
        for file_name, link in dataset_links.items():
            start_time = time.time()

            # Checks if the file already exists. If yes, then does not download the file.
            file_path = os.path.join(dataset_directory_path, f"{file_name}.tgz")
            if os.path.exists(file_path):
                print(f"{file_name}.tgz already exists.")
                print()
                continue

            # Sends request for the current language dataset file.
            response = requests.get(link, stream=True)

            # Checks if the response has a success code. If not then prints the error message.
            assert response.status_code == 200, response.text

            # Gets total file size from headers (in bytes).
            total_size = int(response.headers.get("content-length", 0))

            # Downloads the file with progress bar.
            with open(file_path, "wb") as out_file, tqdm(
                desc=f"Downloading {file_name}.tgz",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    out_file.write(data)
                    pbar.update(len(data))

            print(
                f"Finished downloading dataset for {file_name} split in {(time.time() - start_time):.3f} sec."
            )
            print()

    def extract_dataset() -> None:
        """Extracts files from the LibriSpeech dataset previously downloaded.

        Extracts files from the LibriSpeech dataset previously downloaded.

        Args:
            None.

        Returns:
            None.
        """
        # Checks if the following directory paths exists.
        raw_data_directory_path = check_directory_path_existence(
            os.path.join("data", "raw_data", "librispeech")
        )
        extracted_data_directory_path = check_directory_path_existence(
            os.path.join("data", "extracted_data", "librispeech")
        )

        # Iterates across file names for dataset splits.
        for file_name in ["test", "validation", "train"]:

            # If file path does not exist, then extracts files from the tar file.
            if not os.path.exists(
                os.path.join(
                    extracted_data_directory_path, file_name, "LibriSpeech", "BOOKS.TXT"
                )
            ):
                # Creates absolute directory path for current file name.
                tar_file_path = os.path.join(
                    raw_data_directory_path, f"{file_name}.tgz"
                )

                # Extracts files from downloaded data tar file into a directory.
                try:
                    file = tarfile.open(tar_file_path)
                    file.extractall(
                        os.path.join(extracted_data_directory_path, file_name)
                    )
                    file.close()
                except FileNotFoundError as error:
                    raise FileNotFoundError(f"{tar_file_path} does not exist")
                print(
                    f"Finished extracting files from '{file_name}.tgz' to {extracted_data_directory_path}."
                )

            else:
                print(
                    f"Files for '{file_name}' already exist in {extracted_data_directory_path}. Skipping extraction."
                )
            print()

    def load_dataset_file_paths(self, split_name: str) -> None:
        """Loads file paths and transcription texts for a given dataset split (train, validation & test).

        Loads file paths and transcription texts for a given dataset split (train, validation & test).

        Args:
            split_name: A string for the name of the current dataset split.

        Returns:
            None.
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
                    file_name, text = file_info.split(" ", 1)

                    # Converts characters in text into lowercase, and removes leading & trailing whitespaces.
                    text = text.lower()
                    text = text.strip()

                    # Appends absolute file path & transcription text for current file into dataset info.
                    self.dataset_info[split_name]["file_path"].append(
                        os.path.join(
                            extracted_data_directory_path,
                            dir_0,
                            dir_1,
                            f"{file_name}.flac",
                        )
                    )
                    self.dataset_info[split_name]["text"].append(text)

        print(
            f"No. of examples in the {split_name} data split: {len(self.dataset_info[split_name]['file_path'])}"
        )

    def shuffle_slice_dataset(self) -> None:
        """Zips file paths & transcriptions into single tensor dataset & slices them based on batch size.

        Zips file paths & transcriptions into single tensor dataset, & slices them based on batch size.

        Args:
            None.

        Returns:
            None.
        """
        # Zips file paths & transcriptions into single tensor, and shuffles it.
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.dataset_info["train"]["file_path"],
                self.dataset_info["train"]["text"],
            )
        )
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.dataset_info["validation"]["file_path"],
                self.dataset_info["validation"]["text"],
            )
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.dataset_info["test"]["file_path"],
                self.dataset_info["test"]["text"],
            )
        )

        # Slices the combined dataset based on batch size, and drops remainder values.
        self.batch_size = self.model_configuration["model"]["batch_size"]
        self.train_dataset = self.train_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.validation_dataset = self.validation_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.test_dataset = self.test_dataset.batch(
            self.batch_size, drop_remainder=True
        )

        # Computes number of steps per epoch for all dataset.
        self.n_train_steps_per_epoch = (
            len(self.dataset_info["train"]["file_path"]) // self.batch_size
        )
        self.n_validation_steps_per_epoch = (
            len(self.dataset_info["validation"]["file_path"]) // self.batch_size
        )
        self.n_test_steps_per_epoch = (
            len(self.dataset_info["test"]["file_path"]) // self.batch_size
        )

        print(f"No. of train steps per epoch: {self.n_train_steps_per_epoch}")
        print(f"No. of validation steps per epoch: {self.n_validation_steps_per_epoch}")
        print(f"No. of test steps per epoch: {self.n_test_steps_per_epoch}")
        print()

    def train_tokenizer(self) -> None:
        """Trains a simple character-level tokenizer for CTC-based speech recognition.

        Trains a simple character-level tokenizer for CTC-based speech recognition.

        Args:
            None.

        Returns:
            None.
        """
        # Creates empty dictionary to store char <-> ids.
        self.char_to_id, self.id_to_char = dict(), dict()

        # Iterates across ASCII lowercase to add them to tokenizer.
        for c_id in range(26):
            self.char_to_id[chr(ord("a") + c_id)] = c_id + 1
            self.id_to_char[c_id + 1] = chr(ord("a") + c_id)

        # Adds space & ' to the char <-> id tokenizers.
        self.char_to_id[" "] = c_id + 2
        self.id_to_char[c_id + 2] = " "
        self.char_to_id["'"] = c_id + 3
        self.id_to_char[c_id + 3] = "'"
