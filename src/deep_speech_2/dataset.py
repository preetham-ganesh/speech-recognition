import os

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
