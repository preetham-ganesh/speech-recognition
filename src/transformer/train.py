import os

import mlflow
import tensorflow as tf

from src.utils import load_json_file, check_directory_path_existence
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

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        ]
    )
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
