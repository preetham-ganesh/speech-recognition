import tensorflow as tf

from typing import Dict, Any, List


class DeepSpeech2(tf.keras.Model):
    """A tensorflow model for recognizing text in speech using DeepSpeech2 architecture."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Initializes the DeepSpeech2 model, by adding various layers.

        Initializes the DeepSpeech2 model, by adding various layers.

        Args:
            model_configuration: A dictionary for the configuration of the model.

        Returns:
            None.
        """
        super(DeepSpeech2, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initializes class variables.
        self.model_configuration = model_configuration
        self.cnn_blocks = 2
        self.model_layers = dict()

        # Initializes a multiple Conv Blocks, each with Conv2D, BatchNormalization, & ReLU activation layers.
        kernel_sizes = [(11, 41), (11, 21)]
        stride_vals = [(2, 2), (1, 2)]
        for b_id in range(self.cnn_blocks):
            self.model_layers[f"block_{b_id}_conv2d_0"] = tf.keras.layers.Conv2D(
                filters=model_configuration["model"]["conv_filters"],
                kernel_size=kernel_sizes[b_id],
                strides=stride_vals[b_id],
                padding="same",
                use_bias=False,
                name=f"block_{b_id}_conv2d_0",
            )
            self.model_layers[f"block_{b_id}_bn_0"] = (
                tf.keras.layers.BatchNormalization(name=f"block_{b_id}_bn_0")
            )
            self.model_layers[f"block_{b_id}_relu_0"] = tf.keras.layers.ReLU(
                name=f"block_{b_id}_relu_0"
            )

        # Initialies a Reshape layer to reshape the output from Conv Blocks.
        self.model_layers["reshape_0"] = tf.keras.layers.Reshape(
            target_shape=(
                -1,
                int(
                    tf.math.ceil(self.model_configuration["model"]["n_mels"] / 4)
                    * model_configuration["model"]["conv_filters"]
                ),
            ),
            name="reshape_0",
        )

        # Initializes multiple Bidirectional LSTM blocks.
        for b_id in range(model_configuration["model"]["rnn_blocks"]):
            self.model_layers[f"block_{b_id}_bi_rnn"] = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=model_configuration["model"]["rnn_units"],
                    return_sequences=True,
                    name=f"block_{b_id}_rnn_fwd",
                ),
                merge_mode="concat",
                name=f"block_{b_id}_bi_rnn",
            )
            self.model_layers[f"block_{b_id}_dropout_0"] = tf.keras.layers.Dropout(
                rate=model_configuration["model"]["rate"],
                name=f"block_{b_id}_dropout_0",
            )

        # Initializes the final dense layers.
        self.model_layers["dense_0"] = tf.keras.layers.Dense(
            units=model_configuration["model"]["rnn_units"],
            activation="relu",
            name="dense_0",
        )
        self.model_layers["dropout_0"] = tf.keras.layers.Dropout(
            rate=model_configuration["model"]["rate"], name="dropout_0"
        )
        self.model_layers["final"] = tf.keras.layers.Dense(
            model_configuration["model"]["vocab_size"], name="final"
        )

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> List[tf.Tensor]:
        """Inputs are passed through the layers in the model.

        Inputs are passed through the layers in the model.

        Args:
            inputs: A list of input tensors from the input batch.
            training: A boolean value for the flag of training/testing state.

        Returns:
            A tensors for the output predicted by the model for the current inputs.
        """
        x = inputs[0]

        # Passes the inputs through the Conv blocks.
        for b_id in range(self.cnn_blocks):
            x = self.model_layers[f"block_{b_id}_conv2d_0"](x)
            x = self.model_layers[f"block_{b_id}_bn_0"](x)
            x = self.model_layers[f"block_{b_id}_relu_0"](x)
            print(b_id, x.shape)

        # Passes the Conv block output through the Reshape layer.
        x = self.model_layers["reshape_0"](x)

        # Passes the Reshaped Conv block output (or RNN block output) through the RNN blocks.
        for b_id in range(self.model_configuration["model"]["rnn_blocks"]):
            x = self.model_layers[f"block_{b_id}_bi_rnn"](x)
            x = self.model_layers[f"block_{b_id}_dropout_0"](x, training=training)

        # Passes the RNN block output through the final layers.
        x = self.model_layers["dense_0"](x)
        x = self.model_layers["dropout_0"](x, training=training)
        x = self.model_layers["final"](x)
        return x

    def build_graph(self, max_spectrogram_length: int, n_mels: int) -> tf.keras.Model:
        """Builds plottable graph for the model.

        Builds plottable graph for the model.

        Args:
            max_spectrogram_length: An integer for the maximum no. of time frames in a spectrogram.
            n_mels: An integer for the no. of mel frequency bins to compute.

        Returns:
            A tensorflow keras model for the current model configuration.
        """
        inputs = tf.keras.layers.Input(
            shape=(
                self.model_configuration["model"]["max_spectrogram_length"],
                self.model_configuration["model"]["n_mels"],
                1,
            ),
            name="input_0",
        )
        return tf.keras.Model(inputs=inputs, outputs=self.call([inputs], False))
