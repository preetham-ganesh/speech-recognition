import os
import sys
import warnings
import logging


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


import pandas as pd
import editdistance
import mlflow
import numpy as np

from src.transformer.predict import SpeechRecognition
from src.utils import check_directory_path_existence


class Metrics(object):
    """Computes and logs evaluation metrics for Automatic Speech Recognition (ASR) models."""

    def __init__(
        self,
        d_units: int,
        n_layers: int,
        dataset_size: str,
        representation: str,
    ) -> None:
        """Creates object attributes for the Metrics class.

        Args:
            d_units: An integer for the model's embedding dimension. Must be one of [128, 256, 512, 1024].
            n_layers: An integer for the no. of encoder-decoder layers in the Transformer. Must be between 1 and 6.
            dataset_size: A string indicating the dataset used. Must be either 'mini' or 'full'.
            representation: A string specifying the input representation used, either 'stft' or 'spectrogram'.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(d_units, int) and d_units in [
            128,
            256,
            512,
            1024,
        ], "Variable d_units of type 'int' and should have values as 128, 256, 512 or 1024."
        assert (
            isinstance(n_layers, int) and 0 < n_layers <= 6
        ), "Variable n_layers of type 'int' and should be between 1 & 6."
        assert isinstance(dataset_size, str) and dataset_size in [
            "mini",
            "full",
        ], "Variable dataset_size of type 'str' and should have value as 'mini' or 'full'."
        assert isinstance(representation, str) and representation in [
            "stft",
            "spectrogram",
        ], "Variable representation of type 'str' and should have value as 'stft' or 'spectrogram'."

        # Initalizes class variables.
        self.dataset_size = dataset_size
        self.representation = representation
        self.d_units = d_units
        self.n_layers = n_layers
        self.model_version = f"v-{dataset_size}-{d_units}-{n_layers}-{representation}"

    def load_speech_recognizer(self) -> None:
        """Initializes and loads the speech recognizer model and its configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Creates object attributes for the SpeechRecognition class.
        self.speech_recognizer = SpeechRecognition(
            self.d_units, self.n_layers, self.dataset_size, self.representation
        )

        # Loads the model configuration file for model version.
        self.speech_recognizer.load_model_configuration()

        # Loads model & other utilities for prediction.
        self.speech_recognizer.load_model()

    def load_data(self, split_name: str) -> None:
        """Loads the processed dataset information for a specified data split.

        Args:
            split_name: A string for the name of the split the text belongs to.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(split_name, str), "Variable split_name of type 'str'."

        # Adds split name to class variable.
        self.split_name = split_name

        # Checks if the following directory path exists.
        processed_data_directory_path = check_directory_path_existence(
            os.path.join(
                "data",
                "processed_data",
                "librispeech",
                f"v{self.speech_recognizer.model_configuration['dataset']['version']}",
            )
        )

        # Loads the processed dataset info for current split.
        self.dataset_info = pd.read_csv(
            os.path.join(
                processed_data_directory_path, self.split_name, "dataset_info.csv"
            )
        )
        print(
            f"No. of examples in the {self.split_name} data split: {len(self.dataset_info['validation'])}"
        )
        print()

    def generate_predictions(self) -> None:
        """Generates transcriptions for all examples in the loaded dataset split.

        Args:
            None.

        Returns:
            None.
        """
        # Iterates through each example in the dataset.
        self.predicted_lines = list()
        for f_id in range(len(self.dataset_info)):

            # Predicts the transcription of a given features file using the trained Transformer model.
            predicted_text = self.speech_recognizer.predict(
                str(self.dataset_info["file_path"].iloc[f_id])
            )
            print(f"File path: {str(self.dataset_info['file_path'].iloc[f_id])}")
            print(f"Target text: {self.dataset_info['text']}")
            print(f"Predicted text: {predicted_text}")
            print()

            # Appends predicted text to list.
            self.predicted_lines.append(predicted_text)

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """Computes the Word Error Rate (WER) between a reference and predicted transcription.

        Args:
            reference: A string for the ground truth transcription.
            hypothesis: A string for the predicted transcription from the model.

        Returns:
            A floating value for the computed WER using Levenshtein distance.
        """
        # Asserts type & value of the arguments.
        assert isinstance(reference, str), "Variable reference of type 'str'."
        assert isinstance(hypothesis, str), "Variable hypothesis of type 'str'."

        # Reference & hypothesis into words.
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()

        # Computes word error rate.
        return editdistance.eval(ref_words, hyp_words) / max(1, len(ref_words))

    def compute_cer(self, reference: str, hypothesis: str) -> float:
        """Computes the Character Error Rate (CER) between a reference and predicted transcription.

        Args:
            reference: A string for the ground truth transcription.
            hypothesis: A string for the predicted transcription from the model.

        Returns:
            A floating value for the computed CER using Levenshtein distance.
        """
        # Asserts type & value of the arguments.
        assert isinstance(reference, str), "Variable reference of type 'str'."
        assert isinstance(hypothesis, str), "Variable hypothesis of type 'str'."

        # Computes character error rate.
        return editdistance.eval(reference, hypothesis) / max(1, len(reference))

    def compute_metrics(self) -> None:
        """Computes evaluation metrics (WER and CER) for the current dataset split.

        Args:
            None.

        Returns:
            None.
        """
        # Iterate over all transcriptions in the dataset.
        wer_score, cer_score = list(), list()
        for f_id in range(len(self.dataset_info["text"])):

            # Appends computed WER & CER score to the lists.
            wer_score.append(
                self.compute_wer(
                    str(self.dataset_info["text"].iloc[f_id]),
                    self.predicted_lines[f_id],
                )
            )
            cer_score.append(
                self.compute_cer(
                    str(self.dataset_info["text"].iloc[f_id]),
                    self.predicted_lines[f_id],
                )
            )

        # Log average WER and CER metrics to MLflow, tagged by data split.
        mlflow.log_metrics({f"{self.split_name}_wer": np.mean(wer_score)})
        mlflow.log_metrics({f"{self.split_name}_cer": np.mean(cer_score)})
