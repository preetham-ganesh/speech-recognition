import os
import time

import requests
from tqdm import tqdm

from src.utils import check_directory_path_existence

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
