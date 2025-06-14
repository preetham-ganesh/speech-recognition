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
import librosa

from src.utils import load_json_file, check_directory_path_existence
from src.preprocess_data import load_preprocess_audio
from src.deep_speech_2.model import DeepSpeech2  # your model class

from typing import List, Dict, Any


class DeepSpeech2Predictor(object):
    """Predicts text transcription from audio using DeepSpeech2 model."""

    def __init__(self, model_version: str) -> None:
        assert isinstance(
            model_version, str
        ), "Variable model_version should be of type 'str'."
        self.model_version = model_version

    def load_model_configuration(self) -> None:
        """Loads the model configuration."""
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = os.path.join(
            self.home_directory_path, "configs/deep_speech_2"
        )
        self.model_configuration = load_json_file(
            f"v{self.model_version}", model_configuration_directory_path
        )

    def load_model(self) -> None:
        """Loads the trained DeepSpeech2 model."""

        exported_model = tf.saved_model.load(
            os.path.join(
                self.home_directory_path,
                f"models/deep_speech_2/v{self.model_version}/serialized",
            )
        )

        # Get the callable signature (default is "serving_default")
        self.model = exported_model.signatures["serving_default"]

    def load_preprocess_audio(self, audio_file_path: str) -> tf.Tensor:
        """Loads and preprocesses audio into model input."""
        assert isinstance(audio_file_path, str), "audio_file_path should be str."

        """# Load audio
        signal, sr = librosa.load(audio_file_path, sr=16000)

        # Compute log Mel spectrogram
        spectrogram = compute_log_mel_spectrogram(
            signal, sr, self.model_configuration["model"]["n_mels"]
        )"""

        # Normalize spectrogram to [0, 1]
        spectrogram = load_preprocess_audio(
            audio_file_path, self.model_configuration["model"]["n_mels"]
        )

        # Pad or truncate to max_spectrogram_length
        max_len = self.model_configuration["model"]["max_spectrogram_length"]
        if spectrogram.shape[0] > max_len:
            spectrogram = spectrogram[:max_len, :]
        else:
            pad_width = max_len - spectrogram.shape[0]
            spectrogram = np.pad(
                spectrogram,
                ((0, pad_width), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Add batch and channel dimension
        model_input = tf.expand_dims(spectrogram, axis=0)
        model_input = tf.expand_dims(model_input, axis=-1)
        print(model_input.shape)
        return model_input

    def greedy_decode(self, logits: np.ndarray) -> str:
        """Performs greedy decoding on model output."""
        # logits shape: (batch_size, time_steps, vocab_size)
        print(logits.shape)
        decoded_indices = np.argmax(logits, axis=-1)[0]  # take first sample in batch

        # CTC decoding: remove duplicate tokens and blanks (assume blank=0)
        prev_token = None
        decoded_tokens = []
        for token in decoded_indices:
            if token != prev_token and token != 0:
                decoded_tokens.append(token)
            prev_token = token

        # Map tokens to characters
        idx_to_char = self.model_configuration["tokenizer"][
            "id_to_char"
        ]  # provide this in config
        transcription = "".join([idx_to_char[str(token)] for token in decoded_tokens])
        return transcription

    def predict(self, audio_file_path: str) -> None:
        """Predicts transcription for the audio file."""
        assert isinstance(audio_file_path, str), "audio_file_path should be str."

        # Preprocess audio
        model_input = self.load_preprocess_audio(audio_file_path)

        # Predict
        prediction = self.model(model_input)

        print(prediction["output_0"].shape)

        # Decode prediction
        transcription = self.greedy_decode(prediction["output_0"].numpy())
        print(transcription)

        # Save transcription
        """predictions_dir = check_directory_path_existence(
            f"models/deep_speech_2/v{self.model_version}/predictions"
        )
        n_predictions = len(
            [name for name in os.listdir(predictions_dir) if name.endswith(".txt")]
        )
        output_path = os.path.join(
            predictions_dir, f"{n_predictions}_transcription.txt"
        )
        with open(output_path, "w") as f:
            f.write(transcription)

        print(f"Transcription saved at {output_path}")
        print(f"Transcription: {transcription}")
        print()"""


def main():
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mv",
        "--model_version",
        type=str,
        required=True,
        help="Version of the model used to perform prediction.",
    )
    parser.add_argument(
        "-afp",
        "--audio_file_path",
        type=str,
        required=True,
        help="Location of the audio file.",
    )
    args = parser.parse_args()

    predictor = DeepSpeech2Predictor(args.model_version)

    predictor.load_model_configuration()
    predictor.load_model()
    predictor.predict(args.audio_file_path)


if __name__ == "__main__":
    main()
