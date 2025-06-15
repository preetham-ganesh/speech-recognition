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

    def compute_decoder_embedding(self, x: tf.Tensor) -> None:
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
