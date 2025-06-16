import os

import pandas as pd
import tensorflow as tf
import numpy as np

from src.utils import check_directory_path_existence

from typing import Dict, Any, List


class Dataset(object):
    """Loads the dataset based on the model configuration."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Creates object attributes for the Dataset class.

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
        )
        print(
            f"No. of examples in the train data split: {len(self.dataset_info['train'])}"
        )

        # Loads the processed dataset info for validation split.
        self.dataset_info["validation"] = pd.read_csv(
            os.path.join(
                processed_data_directory_path, "validation", "dataset_info.csv"
            )
        )
        print(
            f"No. of examples in the validation data split: {len(self.dataset_info['validation'])}"
        )

        # Loads the processed dataset info for test split.
        self.dataset_info["test"] = pd.read_csv(
            os.path.join(processed_data_directory_path, "test", "dataset_info.csv")
        )
        print(
            f"No. of examples in the test data split: {len(self.dataset_info['test'])}"
        )
        print()

    def shuffle_slice_dataset(self) -> None:
        """Zips file paths & transcriptions into single tensor dataset & slices them based on batch size.

        Args:
            None.

        Returns:
            None.
        """
        # Zips file paths & transcriptions into single tensor, and shuffles it.
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(self.dataset_info["train"]["file_path"]),
                list(self.dataset_info["train"]["text"]),
            )
        )
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(self.dataset_info["validation"]["file_path"]),
                list(self.dataset_info["validation"]["text"]),
            )
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(self.dataset_info["test"]["file_path"]),
                list(self.dataset_info["test"]["text"]),
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

    def tokenize_text(self, text: str) -> List[int]:
        """Tokenizes text to convert into ids using trained tokenizer.

        Args:
            text: A string for the target text that should be tokenized.

        Returns:
            A list of integers for the tokenized & encoded version of the text.
        """
        # Checks types & values of arguments.
        assert isinstance(text, str), "Variable text should be of type 'str'."

        # Tokenizes characters into ids based on trained tokenizer.
        return [self.char_to_id[c] for c in text]

    def load_input_target_batches(
        self, file_paths: List[str], texts: List[str]
    ) -> List[tf.Tensor]:
        """Loads and preprocesses a batch of audio files and corresponding text labels.

        Args:
            file_paths: A list of strings for locations of audio files in current batch.
            texts: A list of strings for transcriptions of audio files in current batch.

        Returns:
            A list of tensors for input & target batches, and target lengths of spectrograms & tokenized texts.
        """
        # Checks types & values of arguments.
        assert isinstance(
            file_paths, list
        ), "Variable file_paths should be of type 'list'."
        assert isinstance(texts, list), "Variable texts should be of type 'list'."

        # Creates an empty numpy array to store padded versions of STFTs for all audio files in current batch.
        input_batch = np.zeros(
            shape=(
                len(file_paths),
                self.model_configuration["model"]["max_input_length"],
                self.model_configuration["model"]["n_bins"],
            )
        )

        # Creates empty list to store tokenized texts.
        target_batch = list()

        # Iterates across file paths in current batch.
        for f_id in range(len(file_paths)):

            # Loads previously saved STFT for current audio file.
            stfts = np.load(str(file_paths[f_id], "UTF-8"))

            # Pads the loaded STFT based on max_input_length
            pad_amount = max(
                0,
                stfts.shape[0] - self.model_configuration["model"]["max_input_length"],
            )
            stfts = np.pad(stfts, ((0, pad_amount), (0, 0)), mode="constant")[
                : self.model_configuration["model"]["max_input_length"], :
            ]

            # Tokenizes text to convert into ids using trained tokenizer.
            sequence = self.tokenize_text(str(texts[f_id], "UTF-8"))

            # Appends extracted STFT & tokenized & encoded version of text for current file into list.
            input_batch[f_id, :, :] = stfts
            target_batch.append(sequence)

        # Adds extra dimension to the input batch, input & target lengths.
        input_batch = tf.expand_dims(input_batch, axis=-1)

        # Pads input & target batch tensors with 0 at the end.
        target_batch = tf.keras.preprocessing.sequence.pad_sequences(
            target_batch, padding="post", dtype="int32"
        )

        # Converts input & target batches into tensor of data type float32 & int32.
        input_batch = tf.convert_to_tensor(input_batch, dtype=tf.float64)
        input_batch = tf.cast(input_batch, dtype=tf.float32)
        target_batch = tf.convert_to_tensor(target_batch, dtype=tf.int32)
        return [input_batch, target_batch]
