import os

import mlflow
import tensorflow as tf

from src.utils import load_json_file
from src.transformer.dataset import Dataset


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule following an inverse square-root decay strategy with linear warmup."""

    def __init__(self, d_units: int, warmup_steps: int = 4000) -> None:
        """Creates object attributes for the CustomSchedule class.

        Args:
            d_units: An integer for the dimensionality of model's embeddings.
            warmup_steps: An integer for the no. of steps of linear warmup.

        Returns:
            None.
        """
        super(CustomSchedule, self).__init__()

        # Asserts type & values of the arguments.
        assert isinstance(d_units, int), "Variable d_units should be of type 'int'."
        assert isinstance(
            warmup_steps, int
        ), "Variable warmup_steps should be of type 'int'."

        # Initalizes class variables.
        self.d_units = tf.cast(d_units, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        """Computes the learning rate for a given training step.

        Args:
            step: A tensor for current training step.

        Returns:
            A tensor for the computed learning rate at given training step
        """
        # Converts step to float32 to prevent dtype mismatch
        step = tf.cast(step, tf.float32)

        # Computes the inverse square root of the current step & the scaled learning rate for the warmup phase
        arg_1 = tf.math.rsqrt(step)
        arg_2 = step * (self.warmup_steps**-1.5)

        # Scales by the inverse square root of the model dimension and takes the minimum.
        return tf.math.rsqrt(tf.cast(self.d_units, tf.float32)) * tf.math.minimum(
            arg_1, arg_2
        )


class Train(object):
    """Trains the ASR Transformer model based on the configuration."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the Train class.

        Args:
            model_version: A string for the version of the current model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version
        self.best_validation_loss = None

    def load_model_configuration(self) -> None:
        """Loads the model configuration file for model version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = os.path.join(
            self.home_directory_path, "configs", "transformer"
        )
        self.model_configuration = load_json_file(
            f"v{self.model_version}", model_configuration_directory_path
        )

        # Sets tag in MLFlow.
        mlflow.set_tag(
            "architecture", self.model_configuration["model"]["architecture"]
        )
        mlflow.set_tag("dataset_size", self.model_configuration["dataset"]["size"])
        mlflow.set_tag(
            "dataset_version", self.model_configuration["dataset"]["version"]
        )
        mlflow.set_tag("model_type", self.model_configuration["model"]["type"])

        # Logs parameters in MLFlow.
        mlflow.log_param("n_layers", self.n_layers)
        mlflow.log_param("d_units", self.d_units)

    def load_dataset(self) -> None:
        """Loads audio file paths & transcriptions in the dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Creates object attributes for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

        # Loads file paths and transcription texts for the LibriSpeech dataset.
        self.dataset.load_dataset_info()

        # Zips file paths & transcriptions into single tensor dataset & slices them based on batch size.
        self.dataset.shuffle_slice_dataset()

        # Trains a simple character-level tokenizer for CTC-based speech recognition.
        self.dataset.train_tokenizer()

        # Adds trained tokenizer to model configuration.
        self.model_configuration["tokenizer"] = dict()
        self.model_configuration["tokenizer"]["char_to_id"] = self.dataset.char_to_id
        self.model_configuration["tokenizer"]["id_to_char"] = self.dataset.id_to_char

        # Updates model configuration with vocab size.
        self.model_configuration["model"]["vocab_size"] = (
            len(self.dataset.char_to_id) + 1
        )
