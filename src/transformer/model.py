import tensorflow as tf

from typing import Dict, Any, List


class CausalMaskGenerator(tf.keras.layers.Layer):
    """A custom Keras layer that generates a causal (look-ahead) attention mask."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Generates a boolean causal mask based on the input sequence length.

        Args:
            inputs: A tensor for the input of shape (batch_size, target_seq_len, ...)

        Returns:
            A boolean mask tensor where True indicates positions that should be attended to.
        """
        batch_size = tf.shape(inputs)[0]
        target_length = tf.shape(inputs)[1]

        # Creates lower triangular matrix for causal masking
        mask = tf.linalg.band_part(tf.ones((target_length, target_length)), -1, 0)

        # Expands dimensions for batch and convert to bool
        mask = tf.cast(mask, tf.bool)
        mask = tf.expand_dims(mask, 0)  # Add batch dimension
        mask = tf.tile(mask, [batch_size, 1, 1])  # Tile for batch size
        return mask


class Transformer(tf.keras.Model):
    """Speech-to-Text Transformer model with encoder-decoder architecture."""

    def __init__(self, model_configuration: Dict[str, Any]):
        """Initializes the Transformer model, by adding various layers.

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

        # Initializes Conv1D layers for the speech feature embedding.
        self.initialize_encoder_embedding()

        # Initializes the components of a single Transformer encoder layer.
        for l_id in range(self.model_configuration["model"]["n_layers"]):
            self.initialize_encoder_layer(l_id)

        # Initializes decoder embedding layers for mapping target tokens to dense vectors & adding position embedding.
        self.initialize_decoder_embedding()

        # Initializes the components of a single Transformer decoder layer.
        for l_id in range(self.model_configuration["model"]["n_layers"]):
            self.initialize_decoder_layer(l_id)

        # Initializes the Final Dense layer.
        self.model_layers["final"] = tf.keras.layers.Dense(
            units=self.model_configuration["model"]["target_vocab_size"]
        )

    def initialize_encoder_embedding(self) -> None:
        """Initializes Conv1D layers for the speech feature embedding.

        Args:
            None.

        Returns:
            None.
        """
        for l_id in range(3):
            self.model_layers[f"encoder_embedding_conv1d_{l_id}"] = (
                tf.keras.layers.Conv1D(
                    filters=self.model_configuration["model"]["d_units"],
                    kernel_size=11,
                    strides=2,
                    padding="same",
                    activation="relu",
                    name=f"encoder_embedding_conv1d_{l_id}",
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
            x = self.model_layers[f"encoder_embedding_conv1d_{l_id}"](x)
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
        # Computes token embeddings.
        x = self.model_layers["decoder_embedding"](x)

        # Create a custom layer to handle position generation

        class PositionGenerator(tf.keras.layers.Layer):
            def __init__(self, max_length, **kwargs):
                super().__init__(**kwargs)
                self.max_length = max_length

            def call(self, inputs):
                batch_size = tf.shape(inputs)[0]
                seq_length = tf.shape(inputs)[1]
                positions = tf.range(start=0, limit=seq_length, delta=1)
                positions = tf.expand_dims(positions, 0)
                positions = tf.tile(positions, [batch_size, 1])
                return positions

        # Generate positions using the custom layer.
        if not hasattr(self, "_position_generator"):
            self._position_generator = PositionGenerator(
                max_length=self.model_configuration["model"]["target_max_length"]
            )

        # Computes positional embeddings.
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
        ff_output = self.model_layers[f"encoder_{l_id}_dropout_1"](
            ff_output, training=training
        )
        x = self.model_layers[f"encoder_{l_id}_add_1"]([x, ff_output])
        x = self.model_layers[f"encoder_{l_id}_layer_norm_1"](x)
        return x

    def initialize_decoder_layer(self, l_id: int) -> None:
        """Initializes the components of a single Transformer decoder layer.

        Args:
            l_id: An integer for the id of the decoder layer in the Transformer model.

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

    def compute_decoder_attention_mask(
        self, batch_size: int, target_length: int
    ) -> tf.Tensor:
        """Computes a causal attention mask for the Transformer decoder layer to prevent attention to future tokens.

        Args:
            batch_size: An integer for the no. of input & target sequences in current batch.
            target_length: An integer for the length of target sequence.

        Returns:
            A boolean mask tensor where True indicates positions that should be attended to.
        """

        # Creates a custom layer to handle mask generation
        class CausalMaskGenerator(tf.keras.layers.Layer):
            def call(self, inputs):
                batch_size = tf.shape(inputs)[0]
                target_length = tf.shape(inputs)[1]

                # Creates lower triangular matrix for causal masking
                mask = tf.linalg.band_part(
                    tf.ones((target_length, target_length)), -1, 0
                )
                # Expands dimensions for batch and convert to bool
                mask = tf.cast(mask, tf.bool)
                mask = tf.expand_dims(mask, 0)  # Add batch dimension
                mask = tf.tile(mask, [batch_size, 1, 1])  # Tile for batch size
                return mask

        if not hasattr(self, "_mask_generator"):
            self._mask_generator = CausalMaskGenerator()
        return self._mask_generator

    def compute_decoder_output(
        self, l_id: int, x: tf.Tensor, encoder_out: tf.Tensor, training: bool
    ) -> tf.Tensor:
        """Computes the output of a single decoder layer in the Transformer model.

        Args:
            l_id: An integer for the id of the decoder layer in the model.
            x: A tensor for the decoder input from previous decoder layer or decoder embedding.
            encoder_out: A tensor for the output from the last layer in the encoder model.
            training: A boolean value for the flag of training/testing state.

        Returns:
            A tensor for the output computed by the components in the decoder layer in the model.
        """
        # Computes a causal attention mask for the Transformer decoder layer to prevent attention to future tokens.
        causal_mask = None
        if x.shape[0] and x.shape[1]:
            causal_mask = self.compute_decoder_attention_mask(
                x.shape[0], x.shape[1], x.shape[2], tf.bool
            )

        # Computes the multi-head attention layer output for input, and adds output to input as residual connection.
        attention_out_0 = self.model_layers[f"decoder_{l_id}_attention_0"](
            x, x, attention_mask=causal_mask
        )
        attention_out_0 = self.model_layers[f"decoder_{l_id}_dropout_0"](
            attention_out_0, training=training
        )
        x = self.model_layers[f"decoder_{l_id}_add_0"]([x, attention_out_0])
        x = self.model_layers[f"decoder_{l_id}_layer_norm_0"](x)

        # Computes the multi-head attention layer output for encoder out, and adds output to input as residual connection.
        attention_out_1 = self.model_layers[f"decoder_{l_id}_attention_1"](
            x, encoder_out
        )
        attention_out_1 = self.model_layers[f"decoder_{l_id}_dropout_1"](
            attention_out_1, training=training
        )
        x = self.model_layers[f"decoder_{l_id}_add_1"]([x, attention_out_1])
        x = self.model_layers[f"decoder_{l_id}_layer_norm_1"](x)

        # Computes the Feed-forward network layer output, and adds output to attention output as residual connection.
        ff_output = self.model_layers[f"decoder_{l_id}_ffn"](x)
        ff_output = self.model_layers[f"decoder_{l_id}_dropout_1"](
            ff_output, training=training
        )
        x = self.model_layers[f"decoder_{l_id}_add_2"]([x, ff_output])
        x = self.model_layers[f"decoder_{l_id}_layer_norm_1"](x)
        return x

    def call(self, inputs: List[tf.Tensor], training: bool) -> List[tf.Tensor]:
        """Executes the forward pass of the Transformer model.

        Args:
            inputs: A list for input & target sequences representing encoder & decoder inputs.
            training: A boolean value for the flag of training/testing state.

        Returns:
            A list of tensors for the output predicted by the model current inputs.
        """
        input_sequence, target_sequence = inputs

        # Computes the decoder input embedding by combining token and positional embeddings.
        input_sequence = self.compute_encoder_embedding(input_sequence)

        # Computes the output of a single encoder layer in the Transformer model.
        for l_id in range(self.model_configuration["model"]["n_layers"]):
            input_sequence = self.compute_encoder_output(l_id, input_sequence, training)

        # Computes the decoder input embedding by combining token and positional embeddings.
        target_sequence = self.compute_decoder_embedding(target_sequence)

        # Computes the output of a single decoder layer in the Transformer model.
        for l_id in range(self.model_configuration["model"]["n_layers"]):
            input_sequence = self.compute_decoder_output(
                l_id, target_sequence, input_sequence, training
            )

        # Applies final dense layer to map decoder output to target vocabulary size.
        target_sequence = self.model_layers["final"](target_sequence)
        return [target_sequence]

    def build_graph(self) -> tf.keras.Model:
        """"""
        # Defines symbolic input tensors
        input_sequence = tf.keras.layers.Input(
            shape=(None, self.model_configuration["model"]["input_feature_dim"]),
            dtype=tf.float32,
            name="input_sequence",
        )
        target_sequence = tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name="target_sequence"
        )
        return tf.keras.Model(
            inputs=[input_sequence, target_sequence],
            outputs=self.call(inputs=[input_sequence, target_sequence], training=False),
            name="Transformer",
        )
