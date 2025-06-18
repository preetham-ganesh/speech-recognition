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


import tensorflow as tf
import numpy as np

from src.utils import load_json_file
from src.transformer.dataset import Dataset


class SpeechRecognition(object):
    """"""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the SpeechRecognition class.

        Args:
            model_version: A string for the version of the model should be used for prediction.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version

    def load_model_configuration(self) -> None:
        """Loads the model configuration file for model version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = os.path.join(
            self.home_directory_path, "models", "transformer", f"v{self.model_version}"
        )
        self.model_configuration = load_json_file(
            "model_configuration", model_configuration_directory_path
        )

    def load_model(self) -> None:
        """Loads model & other utilities for prediction.

        Args:
            None.

        Returns:
            None.
        """
        # Loads the tensorflow serialized model using model name & version.
        self.home_directory_path = os.getcwd()
        exported_model = tf.saved_model.load(
            os.path.join(
                self.home_directory_path,
                "models",
                "transformer",
                f"v{self.model_version}",
                "serialized",
            )
        )

        # Get the callable signature (default is "serving_default")
        self.model = exported_model.signatures["serving_default"]

        # Initializes object for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

    def load_preprocess_input(self, file_path: str) -> tf.Tensor:
        """Loads & preprocesses STFT of audio file based on model requirements.

        Args:
            file_path: A string for the location of the STFT for the audio file.

        Returns:
            A tensor for the STFT file loaded for the audio file.
        """
        # Asserts type & value of the arguments.
        assert isinstance(file_path, str), "Variable file_path should be of type 'str'."

        # Loads previously saved STFT for current audio file.
        stft = np.load(file_path)

        # Pads the STFT file to maximum input length.
        pad_length = abs(
            self.model_configuration["model"]["max_input_length"] - stft.shape[0]
        )
        stft = np.pad(
            stft,
            pad_width=((0, pad_length), (0, 0)),
            mode="constant",
            constant_values=0,
        )[: self.model_configuration["model"]["max_input_length"], :]

        # Casts STFT to float32 & adds an extra dimensions to it.
        stft = tf.convert_to_tensor(stft, dtype=tf.float32)
        stft = tf.expand_dims(stft, axis=0)
        stft = tf.expand_dims(stft, axis=-1)
        return stft
