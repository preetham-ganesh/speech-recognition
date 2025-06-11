import os

import pandas as pd
import tensorflow as tf

from src.utils import check_directory_path_existence

from typing import Dict, Any, List


class Dataset(object):
    """Loads the dataset based on the model configuration."""

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

    def load_dataset_info(self) -> None:
        """Loads file paths and transcription texts for the LibriSpeech dataset.

        Loads file paths and transcription texts for the LibriSpeech dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Creates an empty dictionary to store dataset info.
        self.dataset_info = dict()

        # Checks if the following directory path exists.
        processed_data_directory_path = check_directory_path_existence(
            os.path.join(
                "data",
                "processed_data",
                "librispeech",
                f"v{self.model_configuration['dataset']['version']}",
            )
        )

        # Loads the processed dataset info for train split.
        self.dataset_info["train"] = pd.read_csv(
            os.path.join(processed_data_directory_path, "train", "dataset_info.csv")
        ).to_dict()
        print(
            f"No. of examples in the train data split: {len(self.dataset_info['train']['file_path'])}"
        )

        # Loads the processed dataset info for validation split.
        self.dataset_info["validation"] = pd.read_csv(
            os.path.join(
                processed_data_directory_path, "validation", "dataset_info.csv"
            )
        ).to_dict()
        print(
            f"No. of examples in the validation data split: {len(self.dataset_info['validation']['file_path'])}"
        )

        # Loads the processed dataset info for test split.
        self.dataset_info["test"] = pd.read_csv(
            os.path.join(processed_data_directory_path, "test", "dataset_info.csv")
        ).to_dict()
        print(
            f"No. of examples in the test data split: {len(self.dataset_info['test']['file_path'])}"
        )
        print()

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
