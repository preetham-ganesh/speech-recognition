import os

import mlflow
import tensorflow as tf

from src.utils import load_json_file, check_directory_path_existence
from src.dataset import Dataset
from src.deep_speech_2 import DeepSpeech2


class Train(object):
    """"""

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

    def load_dataset(self) -> None:
        """Loads audio file paths & transcriptions in the dataset.

        Loads audio file paths & transcriptions in the dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Creates object attributes for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

        # Downloads the LibriSpeech dataset using the OpenSLR links.
        self.dataset.download_dataset()

        # Extracts files from the LibriSpeech dataset previously downloaded.
        self.dataset.extract_dataset()

        # Loads file paths and transcription texts for a given dataset split (train, validation & test).
        self.dataset.load_dataset_file_paths("train")
        self.dataset.load_dataset_file_paths("validation")
        self.dataset.load_dataset_file_paths("test")
        print()

        # Zips file paths & transcriptions into single tensor dataset & slices them based on batch size.
        self.dataset.shuffle_slice_dataset()

        # Trains a simple character-level tokenizer for CTC-based speech recognition.
        self.dataset.train_tokenizer()

        # Updates model configuration with vocab size.
        self.model_configuration["model"]["vocab_size"] = (
            len(self.dataset.char_to_id) + 1
        )

    def load_model(self) -> None:
        """Loads model & other utilies based on model configuration.

        Loads model & other utilies based on model configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Based on model architecture, the model is initialized.
        if self.model_configuration["model"]["architecture"] == "deep_speech_2":
            self.model = DeepSpeech2(
                self.model_configuration["model"]["conv_filters"],
                self.model_configuration["model"]["rnn_units"],
                self.model_configuration["model"]["vocab_size"],
                self.model_configuration["model"]["rnn_blocks"],
                self.model_configuration["model"]["rate"],
            )

        # Builds plottable graph for the model.
        self.model = self.model.build_graph()

        # Loads the optimizer.
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_configuration["model"]["learning_rate"]
        )

        # Creates checkpoint manager for the neural network model.
        self.checkpoint_directory_path = os.path.join(
            self.home_directory_path,
            "models",
            f"v{self.model_version}",
            "checkpoints",
        )
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, model=self.model
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_directory_path, max_to_keep=1
        )
        print("Finished loading model for current configuration.")
        print()

    def generate_model_summary_and_plot(self, plot: bool) -> None:
        """Generates summary & plot for loaded model.

        Generates summary & plot for loaded model.

        Args:
            pool: A boolean value to whether generate model plot or not.

        Returns:
            None.
        """
        # Compiles the model to log the model summary.
        model_summary = list()
        self.model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)
        print(model_summary)
        mlflow.log_text(
            model_summary, os.path.join(f"v{self.model_version}", "model_summary.txt")
        )

        # Creates the following directory path if it does not exist.
        self.reports_directory_path = check_directory_path_existence(
            os.path.join("models", f"v{self.model_version}", "reports")
        )

        # Plots the model & saves it as a PNG file.
        if plot:
            tf.keras.utils.plot_model(
                self.model,
                os.path.join(self.reports_directory_path, "model_plot.png"),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True,
            )

            # Logs the saved model plot PNG file.
            mlflow.log_artifact(
                os.path.join(self.reports_directory_path, "model_plot.png"),
                f"v{self.model_version}",
            )

    def initialize_metric_trackers(self) -> None:
        """Initializes trackers which computes the mean of all metrics.

        Initializes trackers which computes the mean of all metrics.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validation_loss = tf.keras.metrics.Mean(name="validation_loss")
        self.train_cer = tf.keras.metrics.Mean(name="train_cer")
        self.validation_cer = tf.keras.metrics.Mean(name="validation_cer")

    def compute_loss(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes loss for the current batch using actual & predicted values.

        Computes loss for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor for target batch of generated mask images.
            predicted_batch: A tensor for batch of outputs predicted by the model for input batch.

        Returns:
            A tensor for the loss computed on comparing target & predicted batch.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        # Computes loss for current target & predicted batches.
        loss_object = tf.keras.losses.CTC(reduction="sum_over_batch_size", name="ctc")
        current_loss = loss_object(target_batch, predicted_batch)
        return current_loss
