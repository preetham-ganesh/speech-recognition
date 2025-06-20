import os
import time

import mlflow
import tensorflow as tf

from src.utils import load_json_file, check_directory_path_existence, save_json_file
from src.transformer.dataset import Dataset
from src.transformer.model import Transformer


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

    def __init__(
        self,
        d_units: int,
        n_layers: int,
        dataset_size: str,
        dataset_version: str,
        representation: str,
    ) -> None:
        """Creates object attributes for the Train class.

        Args:
            model_version: A string for the version of the current model.

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
        assert isinstance(representation, str) and dataset_size in [
            "stft",
            "spectrogram",
        ], "Variable representation of type 'str' and should have value as 'stft' or 'spectrogram'."

        # Initalizes class variables.
        self.model_version = f"v-{dataset_size}-{d_units}-{n_layers}-{representation}"
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

        # Logs parameters in MLFlow.
        mlflow.log_param("n_layers", self.model_configuration["model"]["n_layers"])
        mlflow.log_param("d_units", self.model_configuration["model"]["d_units"])

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

        # Updates warmup steps in model configuration with n_train_steps_per_epoch.
        self.model_configuration["optimizer"]["warmup_steps"] = (
            self.dataset.n_train_steps_per_epoch * 4
        )

        # Trains a simple character-level tokenizer for CTC-based speech recognition.
        self.dataset.train_tokenizer()

        # Adds trained tokenizer to model configuration.
        self.model_configuration["tokenizer"] = dict()
        self.model_configuration["tokenizer"]["char_to_id"] = self.dataset.char_to_id
        self.model_configuration["tokenizer"]["id_to_char"] = self.dataset.id_to_char

        # Updates model configuration with vocab size.
        self.model_configuration["model"]["target_vocab_size"] = (
            len(self.dataset.char_to_id) + 1
        )

    def load_model(self) -> None:
        """Loads model & other utilies based on model configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Loads model for current model configuration.
        self.model = Transformer(self.model_configuration)

        # Builds plottable graph for the model.
        self.model = self.model.build_graph()

        # Loads the optimizer.
        learning_rate = CustomSchedule(
            self.model_configuration["model"]["d_units"],
            self.model_configuration["optimizer"]["warmup_steps"],
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate,
            beta_1=self.model_configuration["optimizer"]["beta_1"],
            beta_2=self.model_configuration["optimizer"]["beta_2"],
            epsilon=self.model_configuration["optimizer"]["epsilon"],
        )

        # Initializes the loss object using Sparse Categorical Crossentropy.
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

        # Creates checkpoint manager for the neural network model.
        self.checkpoint_directory_path = os.path.join(
            self.home_directory_path,
            "models",
            "transformer",
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
        mlflow.log_text(model_summary, f"v{self.model_version}/model_summary.txt")

        # Creates the following directory path if it does not exist.
        self.reports_directory_path = check_directory_path_existence(
            os.path.join("models", "transformer", f"v{self.model_version}", "reports")
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

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validation_loss = tf.keras.metrics.Mean(name="validation_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
        self.validation_accuracy = tf.keras.metrics.Mean(name="validation_accuracy")

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

        # Computes the loss between the target and predicted values using the loss function
        loss = self.loss_object(target_batch, predicted_batch)

        # Computes a mask to ignore padding tokens (assumed to be represented by 0)
        mask = tf.math.logical_not(tf.math.equal(target_batch, 0))
        mask = tf.cast(mask, dtype=loss.dtype)

        # Applies the mask to the loss to ignore contributions from padding tokens.
        loss *= mask

        # Computes the average loss by summing valid loss values and normalizing by the number of non-padded tokens.
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def compute_accuracy(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes accuracy for the current batch using actual & predicted values.

        Computes accuracy for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor which contains the actual values for the current batch.
            predicted_batch: A tensor which contains the predicted values for the current batch.

        Returns:
            A tensor for the accuracy of current batch.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        # Computes the predicted token indices.
        predicted_batch = tf.argmax(predicted_batch, axis=-1)

        # Ensures dtype compatibility between target batch & predicted batch.
        target_batch = tf.cast(target_batch, dtype=predicted_batch.dtype)

        # Computes element-wise match (excluding padding).
        correct_predictions = tf.equal(target_batch, predicted_batch)

        # Creates mask to ignore padding (assumes 0 is the padding token).
        mask = tf.not_equal(target_batch, 0)

        # Applies mask to correct predictions.
        correct_predictions = tf.cast(correct_predictions & mask, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        # Computes masked accuracy.
        accuracy = tf.reduce_sum(correct_predictions) / tf.reduce_sum(mask)
        return accuracy

    @tf.function
    def train_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Trains model using current input & target batches.

        Args:
            input_batch: A tensor for the input text from the current batch for training the model.
            target_batch: A tensor for the target text from the current batch for training and validating the model.

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

        # Separates target into input (excludes last token) & real (excludes first token).
        target_batch_inp = target_batch[:, :-1]
        target_batch_real = target_batch[:, 1:]

        # Computes the model output for current batch, and metrics for current model output.
        with tf.GradientTape() as tape:
            predictions = self.model([input_batch, target_batch_inp], training=False)
            loss = self.compute_loss(target_batch_real, predictions)
            accuracy = self.compute_accuracy(target_batch_real, predictions)

        # Computes gradients using loss and model variables.
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Uses optimizer to apply the computed gradients on the combined model variables.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Computes batch metrics and appends it to main metrics.
        self.train_loss(loss)
        self.train_accuracy(accuracy)

    def validation_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Validates model using current input & target batches.

        Args:
            input_batch: A tensor for the input text from the current batch for validating the model.
            target_batch: A tensor for the target text from the current batch for validating the model.

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

        # Separates target into input (excludes last token) & real (excludes first token).
        target_batch_inp = target_batch[:, :-1]
        target_batch_real = target_batch[:, 1:]

        # Computes the model output for current batch, and metrics for current model output.
        predictions = self.model([input_batch, target_batch_inp], training=False)
        loss = self.compute_loss(target_batch_real, predictions)
        accuracy = self.compute_accuracy(target_batch_real, predictions)

        # Computes batch metrics and appends it to main metrics.
        self.validation_loss(loss)
        self.validation_accuracy(accuracy)

    def reset_metrics_trackers(self) -> None:
        """Resets states for trackers before the start of each epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss.reset_state()
        self.validation_loss.reset_state()
        self.train_accuracy.reset_state()
        self.validation_accuracy.reset_state()

    def train_model_per_epoch(self, epoch: int) -> None:
        """Trains the model using train dataset for current epoch.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(epoch, int), "Variable epoch should be of type 'int'."

        # Iterates across batches in the train dataset.
        for batch, (file_paths, texts) in enumerate(
            self.dataset.train_dataset.take(self.dataset.n_train_steps_per_epoch)
        ):
            batch_start_time = time.time()

            # Loads and preprocesses a batch of audio files and corresponding text labels.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                list(file_paths.numpy()), list(texts.numpy())
            )

            # Trains the model using the current input and target batch.
            self.train_step(input_batch, target_batch)
            batch_end_time = time.time()
            print(
                f"Epoch={epoch + 1}, Batch={batch}, Train loss={self.train_loss.result().numpy():.3f}, "
                + f"Train accuracy={self.train_accuracy.result().numpy():.3f}, "
                + f"Time taken={(batch_end_time - batch_start_time):.3f} sec."
            )

        # Logs train metrics for current epoch.
        mlflow.log_metrics(
            {
                "train_loss": self.train_loss.result().numpy(),
                "train_accuracy": self.train_accuracy.result().numpy(),
            },
            step=epoch,
        )
        print()

    def validate_model_per_epoch(self, epoch: int) -> None:
        """Validates the model using the current validation dataset.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None
        """
        # Asserts type & value of the arguments.
        assert isinstance(epoch, int), "Variable epoch should be of type 'int'."

        # Iterates across batches in the validation dataset.
        for batch, (file_paths, texts) in enumerate(
            self.dataset.validation_dataset.take(
                self.dataset.n_validation_steps_per_epoch
            )
        ):
            batch_start_time = time.time()

            # Loads and preprocesses a batch of audio files and corresponding text labels.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                list(file_paths.numpy()), list(texts.numpy())
            )

            # Validates the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)
            batch_end_time = time.time()
            print(
                f"Epoch={epoch + 1}, Batch={batch}, Validation loss={self.validation_loss.result().numpy():.3f}, "
                + f"Validation accuracy={self.validation_accuracy.result().numpy():.3f}, "
                + f"Time taken={(batch_end_time - batch_start_time):.3f} sec."
            )

        # Logs train metrics for current epoch.
        mlflow.log_metrics(
            {
                "validation_loss": self.validation_loss.result().numpy(),
                "validation_accuracy": self.validation_accuracy.result().numpy(),
            },
            step=epoch,
        )
        print()

    def save_model(self) -> None:
        """Saves the model after checking performance metrics in current step.

        Args:
            None.

        Returns:
            None.
        """
        self.manager.save()
        print(f"Checkpoint saved at {self.checkpoint_directory_path}.")

    def early_stopping(self) -> bool:
        """Stops the model from learning further if the performance has not improved from previous step break.

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

        Args:
            None.

        Returns:
            None.
        """
        # Initializes trackers which computes the mean of all metrics.
        self.initialize_metric_trackers()

        # Iterates across epochs for training the neural network model.
        for epoch in range(self.model_configuration["model"]["epochs"]):
            epoch_start_time = time.time()

            # Resets states for trackers before the start of each epoch.
            self.reset_metrics_trackers()

            # Trains the model using batces in the train dataset.
            self.train_model_per_epoch(epoch)

            # Validates the model using batches in the validation dataset.
            self.validate_model_per_epoch(epoch)

            epoch_end_time = time.time()
            print(
                f"Epoch={epoch + 1}, Train loss={self.train_loss.result().numpy():.3f}, "
                + f"Validation loss={self.validation_loss.result().numpy():.3f}, "
                + f"Train accuracy={self.train_accuracy.result().numpy():.3f}, "
                + f"Validation accuracy={self.validation_accuracy.result().numpy():.3f}, "
                + f"Time taken={(epoch_end_time - epoch_start_time):.3f} sec."
            )

            # Stops the model from learning further if the performance has not improved from previous epoch.
            model_training_status = self.early_stopping()
            if not model_training_status:
                print(
                    "Model did not improve after 4th time. Model stopped from training further."
                )
                print()
                break
            print()

    def test_model(self) -> None:
        """Tests the trained model using the test dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Resets states for validation metrics.
        self.reset_metrics_trackers()

        # Restore latest saved checkpoint if available.
        self.checkpoint.restore(self.manager.latest_checkpoint)

        # Iterates across batches in the validation dataset.
        for batch, (file_paths, texts) in enumerate(
            self.dataset.test_dataset.take(self.dataset.n_test_steps_per_epoch)
        ):
            batch_start_time = time.time()

            # Loads and preprocesses a batch of audio files and corresponding text labels.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                list(file_paths.numpy()), list(texts.numpy())
            )

            # Validates the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)
            batch_end_time = time.time()
            print(
                f"Batch={batch}, Test loss={self.validation_loss.result().numpy():.3f}, "
                + f"Test accuracy={self.validation_accuracy.result().numpy():.3f}, "
                + f"Time taken={(batch_end_time - batch_start_time):.3f} sec."
            )
        print()
        print(f"Test loss: {self.validation_loss.result().numpy():.3f}.")
        print(f"Test accuracy: {self.validation_accuracy.result().numpy():.3f}")
        print()

        # Logs test metrics for current epoch.
        mlflow.log_metrics(
            {
                "test_loss": self.validation_loss.result().numpy(),
                "test_accuracy": self.validation_accuracy.result().numpy(),
            }
        )

    def serialize_model(self) -> None:
        """Serializes model as TensorFlow module & saves it as MLFlow artifact.

        Args:
            None.

        Returns:
            None.
        """
        # Generates dummy input data for serialization testing.
        input_sequence = tf.ones(
            [
                2,
                self.model_configuration["model"]["max_input_length"],
                self.model_configuration["model"]["n_bins"],
            ],
            dtype=tf.float32,
        )
        target_sequence = tf.ones([2, 50], dtype=tf.int32)

        # Predicts output for the sample input using the model.
        output_0 = self.model([input_sequence, target_sequence], training=False)

        # Saves the model in TF Saved Model format.
        save_path = check_directory_path_existence(
            os.path.join(
                "models",
                "transformer",
                f"v{self.model_version}",
                "serialized",
            )
        )
        self.model.export(save_path)

        # Loads the serialized model to check if the loaded model is callable.
        exported_model = tf.saved_model.load(save_path)

        # Get the callable signature (default is "serving_default")
        serving_model = exported_model.signatures["serving_default"]

        # Predicts output for the sample input using the model
        output_1 = serving_model(
            input_sequence=input_sequence, target_sequence=target_sequence
        )

        # Checks if the shape between output from saved & loaded models matches.
        assert (
            output_0.shape == output_1["output_0"].shape
        ), "Shape does not match between the output from saved & loaded models."
        print("Finished serializing model & configuration files.")
        print()

        # Logs serialized model as artifact.
        mlflow.log_artifacts(save_path, f"v{self.model_version}/model")

        # Saves the updated model configuration in the model directory.
        save_json_file(
            self.model_configuration,
            "model_configuration",
            os.path.join("models", "transformer", f"v{self.model_version}"),
        )

        # Logs updated model configuration as artifact.
        mlflow.log_dict(
            self.model_configuration,
            f"v{self.model_version}/model_configuration.json",
        )
