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

        Initializes Conv1D layers for the speech feature embedding.

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

        Computes encoder/speech feature embedding for the STFT audio input.

        Args:
            x: A tensor for the STFT audio input.

        Returns:
            A tensor for the encoder/speech feature embedding computed.
        """
        for l_id in range(3):
            x = self.model_layers[f"encoder_embedding_conv2d_{l_id}"](x)
        return x
