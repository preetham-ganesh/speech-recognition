import os

import tensorflow as tf

from src.utils import check_directory_path_existence, load_text_file

from typing import Dict, Any


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
