import os
import sys
import warnings
import time


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")


import requests

from src.utils import check_directory_path_existence


def download_dataset() -> None:
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

    # Download the compressed file.
    start_time = time.time()

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
        response = requests.get(link)

        # Checks if the response has a success code. If not then prints the error message.
        assert response.status_code == 200, response.text

        # Saves the compressed file in the response as a .tgz file.
        with open(file_path, "wb") as out_file:
            out_file.write(response.content)
        out_file.close()

        print(
            f"Finished downloading dataset for {file_name} split in {(time.time() - start_time):.3f} sec."
        )
        print()
