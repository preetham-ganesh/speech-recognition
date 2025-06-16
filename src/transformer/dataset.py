import os

import pandas as pd

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
