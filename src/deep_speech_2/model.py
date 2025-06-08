import tensorflow as tf

from typing import List


class DeepSpeech2(tf.keras.Model):
    """A tensorflow model for recognizing text in speech using DeepSpeech2 architecture."""

    def __init__(
        self,
        conv_filters: int,
        rnn_units: int,
        vocab_size: int,
        rnn_blocks: int,
        rate: float,
    ) -> None:
        """Initializes the DeepSpeech2 model, by adding various layers.

        Initializes the DeepSpeech2 model, by adding various layers.

        Args:
            conv_filters: An integer for the no. of filters in each Conv2D layer in the model.
            rnn_units: An integer for the no. of units in each RNN layer.
            vocab_size: An integer for the size of the vocabulary in the language.
            rnn_blocks: An integer for the no. of RNN blocks in the model.
            rate: A floating point value for the dropout rate in the model.

        Returns:
            None.
        """
        super(DeepSpeech2, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(
            conv_filters, int
        ), "Variable conv_filters should be of type 'int'."
        assert isinstance(rnn_units, int), "Variable rnn_units should be of type 'int'."
        assert isinstance(
            vocab_size, int
        ), "Variable vocab_size should be of type of 'int'."
        assert (
            isinstance(rnn_blocks, int) and rnn_blocks > 0
        ), "Variable rnn_blocks should be of type 'int' and value greater than 0."
        assert (
            isinstance(rate, float) and 0 <= rate <= 1
        ), "Variable rate should be of type 'float' and value between 0 & 1 (inclusive)."

        # Initializes class variables.
        self.cnn_blocks = 1
        self.rnn_blocks = rnn_blocks
        self.model_layers = dict()

        # Initializes a multiple Conv Blocks, each with Conv2D, BatchNormalization, & ReLU activation layers.
        kernel_sizes = [(11, 41), (11, 21)]
        stride_vals = [(2, 2), (1, 2)]
        for b_id in range(self.cnn_blocks):
            self.model_layers[f"block_{b_id}_conv2d_0"] = tf.keras.layers.Conv2D(
                filters=conv_filters,
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

        # Initially a Reshape layer to reshape the output from Conv Blocks.
        self.model_layers["reshape_0"] = tf.keras.layers.Lambda(
            lambda t: tf.reshape(
                t, [tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2] * tf.shape(t)[3]]
            ),
            name="reshape_0",
        )

        # Initializes multiple Bidirectional LSTM blocks.
        for b_id in range(self.rnn_blocks):
            self.model_layers[f"block_{b_id}_rnn_fwd"] = tf.keras.layers.LSTM(
                units=rnn_units,
                return_state=True,
                return_sequences=True,
                name=f"block_{b_id}_rnn_fwd",
            )
            self.model_layers[f"block_{b_id}_rnn_bwd"] = tf.keras.layers.LSTM(
                units=rnn_units,
                return_state=True,
                return_sequences=True,
                go_backwards=True,
                name=f"block_{b_id}_rnn_bwd",
            )
            self.model_layers[f"block_{b_id}_bi_rnn"] = tf.keras.layers.Bidirectional(
                layer=self.model_layers[f"block_{b_id}_rnn_fwd"],
                backward_layer=self.model_layers[f"block_{b_id}_rnn_bwd"],
                merge_mode="concat",
                name=f"block_{b_id}_bi_rnn",
            )
            self.model_layers[f"block_{b_id}_dropout_0"] = tf.keras.layers.Dropout(
                rate=rate, name=f"block_{b_id}_dropout_0"
            )

        # Initializes the final dense layers.
        self.model_layers["dense_0"] = tf.keras.layers.Dense(
            units=rnn_units, activation="relu", name="dense_0"
        )
        self.model_layers["dropout_0"] = tf.keras.layers.Dropout(
            rate=rate, name="dropout_0"
        )
        self.model_layers["final"] = tf.keras.layers.Dense(vocab_size, name="final")

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

        # Passes the Conv block output through the Reshape layer.
        x = self.model_layers["reshape_0"](x)

        # Passes the Reshaped Conv block output (or RNN block output) through the RNN blocks.
        for b_id in range(self.rnn_blocks):
            x, *_ = self.model_layers[f"block_{b_id}_bi_rnn"](x)
            x = self.model_layers[f"block_{b_id}_dropout_0"](x, training=training)

        # Passes the RNN block output through the final layers.
        x = self.model_layers["dense_0"](x)
        x = self.model_layers["dropout_0"](x, training=training)
        x = self.model_layers["final"](x)
        return x

    def build_graph(self) -> tf.keras.Model:
        """Builds plottable graph for the model.

        Builds plottable graph for the model.

        Args:
            None.

        Returns:
            A tensorflow keras model for the current model configuration.
        """
        inputs = tf.keras.layers.Input(shape=(None, None, 1), name="input_0")
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs, False))
