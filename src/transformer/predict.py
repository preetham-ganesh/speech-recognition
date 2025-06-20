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

    def __init__(
        self,
        d_units: int,
        n_layers: int,
        dataset_size: str,
        dataset_version: str,
        representation: str,
    ) -> None:
        """Creates object attributes for the SpeechRecognition class.

        Args:
            d_units: An integer for the model's embedding dimension. Must be one of [128, 256, 512, 1024].
            n_layers: An integer for the no. of encoder-decoder layers in the Transformer. Must be between 1 and 6.
            dataset_size: A string indicating the dataset used. Must be either 'mini' or 'full'.
            dataset_version: A string representing the version of the dataset used (e.g., '1.0.0').
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
        assert isinstance(
            dataset_version, str
        ), "Variable dataset_version of type 'str'."
        assert isinstance(representation, str) and representation in [
            "stft",
            "spectrogram",
        ], "Variable representation of type 'str' and should have value as 'stft' or 'spectrogram'."

        # Initalizes class variables.
        self.dataset_size = dataset_size
        self.representation = representation
        self.dataset_version = dataset_version
        self.d_units = d_units
        self.n_layers = n_layers
        self.model_version = f"v-{dataset_size}-{d_units}-{n_layers}-{representation}"

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
        return stft

    def predict(self, file_path: str) -> str:
        """Predicts the transcription of a given STFT file using the trained Transformer model.

        Args:
            file_path: A string for the location where the STFT of the audio file is located.

        Returns:
            A string for the transcription generated by the model.
        """
        # Asserts type & value of the arguments.
        assert isinstance(file_path, str), "Variable text of type 'str'."

        # Loads & preprocesses STFT of audio file based on model requirements.
        stft = self.load_preprocess_input(file_path)

        # Initializes decoder input with the start token of the target language.
        decoder_input = tf.expand_dims(
            [self.model_configuration["tokenizer"]["char_to_id"]["<s>"]], 0
        )

        # Iterates across maximum decoding steps to prevent infinite loops.
        predicted_text = list()
        for _ in range(100):

            # Perform forward pass through the model to get predictions.
            predictions = self.model(input_sequence=stft, target_sequence=decoder_input)

            # Extracts the most probable next token.
            predicted_id = tf.cast(
                tf.argmax(predictions["output_0"][:, -1:, :], axis=-1), tf.int32
            )

            # Appends the predicted token to the decoder input.
            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

            # Stops decoding if the end-of-sequence (EOS) token is generated.
            if (
                self.model_configuration["tokenizer"]["id_to_char"][
                    str(predicted_id.numpy()[0][0])
                ]
                == "</s>"
            ):
                break

            # Appends predicted character into list.
            predicted_text.append(
                self.model_configuration["tokenizer"]["id_to_char"][
                    str(predicted_id.numpy()[0][0])
                ]
            )

        # Returns predicted characters as a single text.
        return "".join(predicted_text)


def main():
    print()

    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mv",
        "--model_version",
        type=str,
        required=True,
        help="Version of the model used to perform the prediction.",
    )
    parser.add_argument(
        "-fp",
        "--file_path",
        type=str,
        required=True,
        help="Location where the STFT of the audio file is located.",
    )
    args = parser.parse_args()

    # Creates object attributes for the SpeechRecognition class.
    speech_recognition = SpeechRecognition(args.model_version)

    # Loads the model configuration file for model version.
    speech_recognition.load_model_configuration()

    # Loads model & other utilities for prediction.
    speech_recognition.load_model()

    # Predicts the transcription of a given STFT file using the trained Transformer model.
    result = speech_recognition.predict(args.file_path)
    print(f"Predicted text: {result}")
    print()


if __name__ == "__main__":
    main()
