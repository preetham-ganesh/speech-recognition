import os
import sys
import warnings
import time
import tarfile


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
    for file_name in ["train", "validation", "test"]:

        # If file path does not exist, then extracts files from the tar file.
        if not os.path.exists(
            os.path.join(extracted_data_directory_path, file_name, "BOOKS.TXT")
        ):
            # Creates absolute directory path for current file name.
            tar_file_path = os.path.join(raw_data_directory_path, f"{file_name}.tgz")

            # Extracts files from downloaded data tar file into a directory.
            try:
                file = tarfile.open(tar_file_path)
                file.extractall(extracted_data_directory_path)
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
