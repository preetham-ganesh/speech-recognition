import os
import sys
import warnings
import logging
import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


import mlflow

from src.utils import set_physical_devices_memory_limit, check_directory_path_existence
from src.transformer.train import Train
from src.transformer.metrics import Metrics


def main():
    print()

    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-en",
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the MLFlow experiment on which the model should be trained.",
    )
    parser.add_argument(
        "-ds",
        "--dataset_size",
        type=str,
        required=True,
        help="Size of the dataset used for training the model.",
    )
    parser.add_argument(
        "-dv",
        "--dataset_version",
        type=str,
        required=True,
        help="Version of the dataset used for training the model.",
    )
    parser.add_argument(
        "-du",
        "--d_units",
        type=int,
        required=True,
        help="No. of units in the Encoder & Decoder layers in the Transformer model.",
    )
    parser.add_argument(
        "-nl",
        "--n_layers",
        type=int,
        required=True,
        help="No. of layers in the Transformer model.",
    )
    parser.add_argument(
        "-r",
        "--representation",
        type=str,
        required=True,
        help="Type of representation to be computed from the audio file.",
    )
    args = parser.parse_args()

    # Sets memory limit of GPU if found in the system.
    set_physical_devices_memory_limit()

    # Creates the directory path.
    mlruns_directory_path = check_directory_path_existence("mlruns")

    # Sets tracking URI for MLFlow server.
    mlflow.set_tracking_uri(mlruns_directory_path)

    # Gets experiment based on name.
    existing_experiment = mlflow.get_experiment_by_name(args.experiment_name)

    # If experiment does not exist, then creates experiment with name & set it.
    if not existing_experiment:
        mlflow.create_experiment(args.experiment_name, artifact_location="")
    mlflow.set_experiment(args.experiment_name)

    # Creates an object for the Train class.
    trainer = Train(
        args.d_units,
        args.n_layers,
        args.dataset_size,
        args.dataset_version,
        args.representation,
    )

    # Loads model configuration for current model version.
    trainer.load_model_configuration()

    # Loads dataset based on dataset version in the model configuration.
    trainer.load_dataset()

    # Loads model & other utilies based on model configuration.
    trainer.load_model()

    # Generates summary & plot for loaded model.
    trainer.generate_model_summary_and_plot(True)

    # Trains & validates the model using train & validation dataset.
    trainer.fit()

    # Tests the model using the test dataset.
    trainer.test_model()

    # Serializes model as TensorFlow module & saves it as MLFlow artifact.
    trainer.serialize_model()

    # Creates object attributes for the Metrics class.
    metrics = Metrics(
        args.d_units, args.n_layers, args.dataset_size, args.representation
    )

    # Initializes and loads the speech recognizer model and its configuration.
    metrics.load_speech_recognizer()

    # Computes evaluation metrics (WER and CER) for the current dataset split.
    metrics.compute_metrics("validation")

    # Computes evaluation metrics (WER and CER) for the current dataset split.
    metrics.compute_metrics("test")


if __name__ == "__main__":
    main()
