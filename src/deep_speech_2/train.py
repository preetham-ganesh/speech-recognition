import os

import mlflow

from src.utils import load_json_file


class Train(object):
    """Trains the DeepSpeech2 model based on the configuration."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the Train class.

        Creates object attributes for the Train class.

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
        self.step = 0

    def load_model_configuration(self) -> None:
        """Loads the model configuration file for model version.

        Loads the model configuration file for model version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = os.path.join(
            self.home_directory_path, "configs"
        )
        self.model_configuration = load_json_file(
            f"v{self.model_version}", model_configuration_directory_path
        )

        # Sets tag in MLFlow.
        mlflow.set_tag(
            "architecture", self.model_configuration["model"]["architecture"]
        )

        # Logs parameters in MLFlow.
        mlflow.log_param(
            "conv_filters", self.model_configuration["model"]["conv_filters"]
        )
        mlflow.log_param("rnn_units", self.model_configuration["model"]["rnn_units"])
        mlflow.log_param("rnn_blocks", self.model_configuration["model"]["rnn_blocks"])
