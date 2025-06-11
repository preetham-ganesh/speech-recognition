import os
import time

import mlflow
import tensorflow as tf

from src.utils import load_json_file, check_directory_path_existence
from src.deep_speech_2.dataset import Dataset
from src.deep_speech_2.model import DeepSpeech2


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
            self.home_directory_path, "configs", "deep_speech_2"
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
            self.model = DeepSpeech2(self.model_configuration)

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
            "deep_speech_2",
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
            os.path.join("models", "deep_speech_2", f"v{self.model_version}", "reports")
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

    def compute_loss(
        self,
        target_batch: tf.Tensor,
        predicted_batch: tf.Tensor,
        target_lengths: tf.Tensor,
    ) -> tf.Tensor:
        """Computes loss for the current batch using actual & predicted values.

        Computes loss for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor for target batch of generated mask images.
            predicted_batch: A tensor for batch of outputs predicted by the model for input batch.
            target_lengths: A tensor for the batch of target sequence length per sample (before padding).

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
        assert isinstance(
            target_lengths, tf.Tensor
        ), "Variable target_lengths should be of type 'tf.Tensor'."

        # Computes predicted lengths using sequences in predicted batch.
        predicted_lengths = [predicted_batch.shape[1]] * predicted_batch.shape[0]

        # Computes loss for current target & predicted batches.
        current_loss = tf.nn.ctc_loss(
            labels=target_batch,
            logits=predicted_batch,
            label_length=target_lengths,
            logit_length=predicted_lengths,
            blank_index=0,
            logits_time_major=False,
        )
        return tf.reduce_mean(current_loss)

    @tf.function
    def train_step(
        self,
        input_batch: tf.Tensor,
        target_batch: tf.Tensor,
        target_lengths: tf.Tensor,
    ) -> None:
        """Trains the model using input & target batches.

        Trains the model using input & target batches.

        Args:
            input_batch: A tensor for input batch of processed images.
            target_batch: A tensor for target batch of generated mask images.
            target_lengths: A tensor for the batch of target sequence length per sample (before padding).

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_lengths, tf.Tensor
        ), "Variable target_lengths should be of type 'tf.Tensor'."

        # Computes predicted audio transcriptions for all audio files in the batch, and computes batch loss.
        with tf.GradientTape() as tape:
            predicted_batch = self.model([input_batch], training=True)
            batch_loss = self.compute_loss(
                target_batch, predicted_batch, target_lengths
            )

        # Computes gradients using loss. Apply the computed gradients on model variables using optimizer.
        gradients = tape.gradient(batch_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Computes mean for loss.
        self.train_loss(batch_loss)

    def validation_step(
        self,
        input_batch: tf.Tensor,
        target_batch: tf.Tensor,
        target_lengths: tf.Tensor,
    ) -> None:
        """Validates the model using input & target batches.

        Validates the model using input & target batches.

        Args:
            input_batch: A tensor for input batch of processed images.
            target_batch: A tensor for target batch of generated mask images.
            target_lengths: A tensor for the batch of target sequence length per sample (before padding).

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_lengths, tf.Tensor
        ), "Variable target_lengths should be of type 'tf.Tensor'."

        # Computes predicted audio transcriptions for all audio files in the batch, and computes batch loss.
        predicted_batch = self.model([input_batch], training=False)
        batch_loss = self.compute_loss(target_batch, predicted_batch, target_lengths)

        # Computes mean for loss.
        self.validation_loss(batch_loss)

    def reset_metrics_trackers(self) -> None:
        """Resets states for trackers before the start of each epoch.

        Resets states for trackers before the start of each epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss.reset_state()
        self.validation_loss.reset_state()

    def validate_model(self) -> None:
        """Validates the model using the current validation dataset.

        Validates the model using the current validation dataset.

        Args:
            None.

        Returns:
            None
        """
        # Iterates across batches in the validation dataset.
        for batch, (file_paths, texts) in enumerate(
            self.dataset.validation_dataset.take(
                self.dataset.n_validation_steps_per_epoch
            )
        ):
            batch_start_time = time.time()

            # Loads and preprocesses a batch of audio files and corresponding text labels.
            input_batch, target_batch, target_lengths = (
                self.dataset.load_input_target_batches(
                    list(file_paths.numpy()), list(texts.numpy())
                )
            )

            # Validates the model using the current input and target batch.
            self.validation_step(input_batch, target_batch, target_lengths)
            batch_end_time = time.time()
            print(
                f"Step={self.step}, Batch={batch}, Validation loss={self.validation_loss.result().numpy():.3f}, "
                + f"Time taken={(batch_end_time - batch_start_time):.3f} sec."
            )

        # Logs train metrics for current epoch.
        mlflow.log_metrics(
            {"validation_loss": self.validation_loss.result().numpy()},
            step=self.step,
        )
        print()

    def save_model(self) -> None:
        """Saves the model after checking performance metrics in current step.

        Saves the model after checking performance metrics in current step.

        Args:
            None.

        Returns:
            None.
        """
        self.manager.save()
        print(f"Checkpoint saved at {self.checkpoint_directory_path}.")

    def early_stopping(self) -> bool:
        """Stops the model from learning further if the performance has not improved from previous step break.

        Stops the model from learning further if the performance has not improved from previous step break.

        Args:
            None.

        Returns:
            None.
        """
        # If epoch = 1, then best validation loss is replaced with current validation loss, & the checkpoint is saved.
        if self.best_validation_loss is None:
            self.patience_count = 0
            self.best_validation_loss = round(
                float(self.validation_loss.result().numpy()), 3
            )
            self.save_model()

        # If best validation loss is higher than current validation loss, the best validation loss is replaced with
        # current validation loss, & the checkpoint is saved.
        elif self.best_validation_loss > round(
            float(self.validation_loss.result().numpy()), 3
        ):
            self.patience_count = 0
            print(
                f"Best validation loss changed from {self.best_validation_loss} to "
                + f"{self.validation_loss.result().numpy():.3f}"
            )
            self.best_validation_loss = round(
                float(self.validation_loss.result().numpy()), 3
            )
            self.save_model()

        # If best validation loss is not higher than the current validation loss, then the number of times the model
        # has not improved is incremented by 1.
        elif self.patience_count < self.model_configuration["model"]["patience_count"]:
            self.patience_count += 1
            print("Best validation loss did not improve.")
            print("Checkpoint not saved.")

        # If the number of times the model did not improve is greater than 4, then model is stopped from training.
        else:
            return False
        return True

    def fit(self) -> None:
        """Trains & validates the loaded model using train & validation dataset.

        Trains & validates the loaded model using train & validation dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes TensorFlow trackers which computes the mean of all metrics.
        self.initialize_metric_trackers()

        # Iterates across epochs for training the neural network model.
        for _ in range(self.model_configuration["model"]["epochs"]):

            # Iterates across batches in the train dataset.
            for batch, (file_paths, texts) in enumerate(
                self.dataset.train_dataset.take(self.dataset.n_train_steps_per_epoch)
            ):
                batch_start_time = time.time()

                # Loads and preprocesses a batch of audio files and corresponding text labels.
                input_batch, target_batch, target_lengths = (
                    self.dataset.load_input_target_batches(
                        list(file_paths.numpy()), list(texts.numpy())
                    )
                )
                print(round(time.time() - batch_start_time, 3))

                # Trains the model using the current input and target batch.
                self.train_step(input_batch, target_batch, target_lengths)
                batch_end_time = time.time()
                print(
                    f"Step={self.step}, Batch={batch}, Train loss={self.train_loss.result().numpy():.3f}, "
                    + f"Time taken={(batch_end_time - batch_start_time):.3f} sec."
                )
                print(round(time.time() - batch_start_time, 3))

                # Logs train metrics for current epoch.
                mlflow.log_metrics(
                    {"train_loss": self.train_loss.result().numpy()},
                    step=self.step,
                )

                # If step is not 0, and step is divisible by step break, then validates the model.
                if (
                    self.step != 0
                    and self.step % self.model_configuration["model"]["step_break"] == 0
                ):
                    print()
                    self.validate_model()
                    print(
                        f"Step={self.step}, Train loss={self.train_loss.result().numpy():.3f}, "
                        + f"Validation loss={self.validation_loss.result().numpy():.3f}"
                    )

                    # Stops the model from learning further if the performance has not improved from previous epoch.
                    model_training_status = self.early_stopping()
                    if not model_training_status:
                        print(
                            "Model did not improve after 4th time. Model stopped from training further."
                        )
                        print()
                        return

                    # Resets states for training and validation metrics before the start of each epoch.
                    self.reset_metrics_trackers()
                    print()

                # If the model training has completed max train steps, then stops the model from training further.
                if self.step == self.model_configuration["model"]["max_steps"]:
                    return
                self.step += 1

    def test_model(self) -> None:
        """Tests the trained model using the test dataset.

        Tests the trained model using the test dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Resets states for validation metrics.
        self.reset_metrics_trackers()

        # Restore latest saved checkpoint if available.
        self.checkpoint.restore(self.manager.latest_checkpoint)

        # Iterates across batches in the test dataset.
        for batch, (file_paths, texts) in enumerate(
            self.dataset.test_dataset.take(self.dataset.n_test_steps_per_epoch)
        ):

            # Loads and preprocesses a batch of audio files and corresponding text labels.
            input_batch, target_batch, target_lengths = (
                self.dataset.load_input_target_batches(
                    list(file_paths.numpy()), list(texts.numpy())
                )
            )

            # Validates the model using the current input and target batch.
            self.validation_step(input_batch, target_batch, target_lengths)

        print(f"Test loss: {self.validation_loss.result().numpy():.3f}.")
        print()

        # Logs test metrics for current epoch.
        mlflow.log_metrics({"test_loss": self.validation_loss.result().numpy()})
