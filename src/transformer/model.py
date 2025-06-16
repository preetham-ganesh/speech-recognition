import tensorflow as tf

from typing import Dict, Any


class Transformer(tf.keras.Model):
    """"""

    def __init__(self, model_configuration: Dict[str, Any]):
        """Initializes the Transformer model, by adding various layers.

        Initializes the Transformer model, by adding various layers.

        Args:
            model_configuration: A dictionary for the configuration of the model.

        Returns:
            None.
        """
        super(Transformer, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initializes class variables.
        self.model_configuration = model_configuration
        self.model_layers = dict()

    def initialize_encoder_embedding(self) -> None:
        """Initializes Conv1D layers for the speech feature embedding.

        Args:
            None.

        Returns:
            None.
        """
        for l_id in range(3):
            self.model_layers[f"encoder_embedding_conv2d_{l_id}"] = (
                tf.keras.layers.Conv1D(
                    filters=self.model_configuration["model"]["d_units"],
                    strides=2,
                    padding="same",
                    activation="relu",
                    name=f"encoder_embedding_conv2d_{l_id}",
                )
            )

    def compute_encoder_embedding(self, x: tf.Tensor) -> tf.Tensor:
        """Computes encoder/speech feature embedding for the STFT audio input.

        Args:
            x: A tensor for the STFT audio input.

        Returns:
            A tensor for the encoder/speech feature embedding computed.
        """
        for l_id in range(3):
            x = self.model_layers[f"encoder_embedding_conv2d_{l_id}"](x)
        return x

    def initialize_decoder_embedding(self) -> None:
        """Initializes decoder embedding layers for mapping target tokens to dense vectors & adding position embedding.

        Args:
            None.

        Returns:
            None.
        """
        self.model_layers["decoder_embedding"] = tf.keras.layers.Embedding(
            input_dim=self.model_configuration["model"]["target_vocab_size"],
            output_dim=self.model_configuration["model"]["d_units"],
            name="decoder_embedding",
        )
        self.model_layers["decoder_positional_embedding"] = tf.keras.layers.Embedding(
            input_dim=self.model_configuration["model"]["target_max_length"],
            output_dim=self.model_configuration["model"]["d_units"],
            name="decoder_positional_embedding",
        )
        self.model_layers["decoder_embedding_add_0"] = tf.keras.layers.Add(
            name="decoder_embedding_add_0"
        )

    def compute_decoder_embedding(self, x: tf.Tensor) -> tf.Tensor:
        """Computes the decoder input embedding by combining token and positional embeddings.

        Args:
            x: A tensor of target token indices with shape (batch_size, sequence_length).

        Returns:
            A tensor representing the combined token and positional embeddings.
        """
        x = self.model_layers["decoder_embedding"](x)
        positions = tf.range(
            start=0,
            limit=self.model_configuration["model"]["target_max_length"],
            delta=1,
        )
        positions = self.model_layers["decoder_positional_embedding"](positions)
        x = self.model_layers["decoder_embedding_add_0"]([x, positions])
        return x

    def initialize_encoder_layer(self, l_id: int) -> None:
        """Initializes components in a single Transformer Encoder layer.

        Args:
            l_id: An integer for the id of the encoder layer in the Transformer model.

        Returns:
            None.
        """
        self.model_layers[f"encoder_{l_id}_attention_0"] = (
            tf.keras.layers.MultiHeadAttention(
                num_heads=self.model_configuration["model"]["n_heads"],
                key_dim=self.model_configuration["model"]["d_units"],
                name=f"encoder_{l_id}_attention_0",
            )
        )
        self.model_layers[f"encoder_{l_id}_ffn"] = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=self.model_configuration["model"]["ff_units"],
                    activation="relu",
                    name=f"encoder_{l_id}_ffn_dense_0",
                ),
                tf.keras.layers.Dense(
                    units=self.model_configuration["model"]["d_units"],
                    name=f"encoder_{l_id}_ffn_dense_1",
                ),
            ]
        )
        self.model_layers[f"encoder_{l_id}_layer_norm_0"] = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6, name=f"encoder_{l_id}_layer_norm_0"
            )
        )
        self.model_layers[f"encoder_{l_id}_layer_norm_1"] = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6, name=f"encoder_{l_id}_layer_norm_1"
            )
        )
        self.model_layers[f"encoder_{l_id}_dropout_0"] = tf.keras.layers.Dropout(
            rate=self.model_configuration["model"]["rate"],
            name=f"encoder_{l_id}_dropout_0",
        )
        self.model_layers[f"encoder_{l_id}_dropout_1"] = tf.keras.layers.Dropout(
            rate=self.model_configuration["model"]["rate"],
            name=f"encoder_{l_id}_dropout_1",
        )
        self.model_layers[f"encoder_{l_id}_add_0"] = tf.keras.layers.Add(
            name=f"encoder_{l_id}_add_0"
        )
        self.model_layers[f"encoder_{l_id}_add_1"] = tf.keras.layers.Add(
            name=f"encoder_{l_id}_add_1"
        )

    def compute_encoder_output(
        self, l_id: int, x: tf.Tensor, training: bool
    ) -> tf.Tensor:
        """Computes the output of a single encoder layer in the Transformer model.

        Args:
            l_id: An integer for the id of the encoder layer in the model.
            x: A tensor for the input from the previous encoder layer or encoder embedding.
            training: A boolean value for the flag of training/testing state.

        Returns:
            A tensor for the output computed by the current encoder layer in the model.
        """
        # Computes the multi-head attention layer output for input, and adds output to input as residual connection.
        attention_out = self.model_layers[f"encoder_{l_id}_attention_0"](x, x)
        attention_out = self.model_layers[f"encoder_{l_id}_dropout_0"](
            attention_out, training=training
        )
        x = self.model_layers[f"encoder_{l_id}_add_0"]([x, attention_out])
        x = self.model_layers[f"encoder_{l_id}_layer_norm_0"](x)

        # Computes the Feed-forward network layer output, and adds output to attention output as residual connection.
        ff_output = self.model_layers[f"encoder_{l_id}_ffn"](x)
        ff_output = self.model_layers[f"encoder_{l_id}_dropout_dropout_1"](
            ff_output, training=training
        )
        x = self.model_layers[f"encoder_{l_id}_add_1"]([x, ff_output])
        x = self.model_layers[f"encoder_{l_id}_layer_norm_1"](x)
        return x

    def initialize_decoder_layer(self, l_id: int) -> None:
        """Initializes the components of a single Transformer decoder layer.

        Args:
            l_id: An integer for the id of the encoder layer in the Transformer model.

        Returns:
            None.
        """
        self.model_layers[f"decoder_{l_id}_attention_0"] = (
            tf.keras.layers.MultiHeadAttention(
                num_heads=self.model_configuration["model"]["n_heads"],
                key_dim=self.model_configuration["model"]["d_units"],
                name=f"decoder_{l_id}_attention_0",
            )
        )
        self.model_layers[f"decoder_{l_id}_attention_1"] = (
            tf.keras.layers.MultiHeadAttention(
                num_heads=self.model_configuration["model"]["n_heads"],
                key_dim=self.model_configuration["model"]["d_units"],
                name=f"decoder_{l_id}_attention_1",
            )
        )
        self.model_layers[f"decoder_{l_id}_ffn"] = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=self.model_configuration["model"]["ff_units"],
                    activation="relu",
                    name=f"decoder_{l_id}_ffn_dense_0",
                ),
                tf.keras.layers.Dense(
                    units=self.model_configuration["model"]["d_units"],
                    name=f"decoder_{l_id}_ffn_dense_1",
                ),
            ]
        )
        self.model_layers[f"decoder_{l_id}_layer_norm_0"] = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6, name=f"decoder_{l_id}_layer_norm_0"
            )
        )
        self.model_layers[f"decoder_{l_id}_layer_norm_1"] = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6, name=f"decoder_{l_id}_layer_norm_1"
            )
        )
        self.model_layers[f"decoder_{l_id}_layer_norm_2"] = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6, name=f"decoder_{l_id}_layer_norm_2"
            )
        )
        self.model_layers[f"decoder_{l_id}_dropout_0"] = tf.keras.layers.Dropout(
            rate=self.model_configuration["model"]["rate"],
            name=f"decoder_{l_id}_dropout_0",
        )
        self.model_layers[f"decoder_{l_id}_dropout_1"] = tf.keras.layers.Dropout(
            rate=self.model_configuration["model"]["rate"],
            name=f"decoder_{l_id}_dropout_1",
        )
        self.model_layers[f"decoder_{l_id}_dropout_2"] = tf.keras.layers.Dropout(
            rate=self.model_configuration["model"]["rate"],
            name=f"decoder_{l_id}_dropout_2",
        )
        self.model_layers[f"decoder_{l_id}_add_0"] = tf.keras.layers.Add(
            name=f"decoder_{l_id}_add_0"
        )
        self.model_layers[f"decoder_{l_id}_add_1"] = tf.keras.layers.Add(
            name=f"decoder_{l_id}_add_1"
        )
