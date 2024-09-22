# import tensorflow as tf
# from tensorflow.keras.layers import Dense, LayerNormalization, Layer, Dropout
# from tensorflow.keras.models import Model

# class PatchEncoder(tf.keras.layers.Layer):
#     def __init__(
#         self, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM, **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.num_patches = num_patches
#         self.position_embedding = tf.keras.layers.Embedding(
#             input_dim=num_patches, output_dim=projection_dim
#         )
#         self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

#     def call(self, encoded_patches):
#         encoded_positions = self.position_embedding(self.positions)
#         encoded_patches = encoded_patches + encoded_positions
#         return encoded_patches


# def mlp(x, hidden_units, dropout_rate):
#     for units in hidden_units:
#         x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
#         x = tf.keras.layers.Dropout(dropout_rate)(x)
#     return x

# class TransformerBlock(Layer):
#     def __init__(self, num_heads, embed_dim, ff_dim, rate=0.1):
#         super(TransformerBlock, self).__init__()
#         self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = tf.keras.Sequential([
#             Dense(ff_dim, activation=tf.nn.gelu),
#             Dense(embed_dim),
#         ])
#         self.layernorm1 = LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = LayerNormalization(epsilon=1e-6)
#         self.dropout1 = Dropout(rate)
#         self.dropout2 = Dropout(rate)

#     def call(self, inputs, training):
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output)

# def create_vit_classifier(input_shape, num_classes, patch_size, num_patches, projection_dim, num_heads, transformer_layers, transformer_units, mlp_head_units, dropout_rate=0.1):
#     inputs = tf.keras.layers.Input(shape=input_shape)
#     patches = tf.image.extract_patches(
#         images=inputs,
#         sizes=[1, patch_size, patch_size, 1],
#         strides=[1, patch_size, patch_size, 1],
#         rates=[1, 1, 1, 1],
#         padding='VALID'
#     )
#     # patches_shape = patches.shape
#     # patch_dim = patches_shape[-1]
#     flat_patches = tf.keras.layers.Reshape((num_patches, -1))(patches)
#     tokens = tf.keras.layers.Dense(units=projection_dim)(flat_patches)
#     encoded_patches = PatchEncoder()(tokens)
#     for _ in range(transformer_layers):
#         x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#         attention_output = tf.keras.layers.MultiHeadAttention(
#                 num_heads=num_heads, key_dim=projection_dim, dropout=0.1
#             )(x1, x1)
#         # Skip connection 1.
#         x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
#         # Layer normalization 2.
#         x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
#         # MLP.
#         TRANSFORMER_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
#         x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
#         # Skip connection 2.
#         encoded_patches = layers.Add()([x3, x2])
#     representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(embedded_patches)
#     representation = tf.keras.layers.Flatten()(representation)
#     representation = Dropout(dropout_rate)(representation)
#     features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
#     logits = Dense(num_classes)(features)
#     return Model(inputs=inputs, outputs=logits)

import math

# import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class PatchTokenization(layers.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        num_patches,
        projection_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.flatten_patches = layers.Reshape((num_patches, -1))
        self.projection = layers.Dense(units=projection_dim)

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        tokens = self.projection(flat_patches)
        return tokens


def create_vit_classifier(
    input_shape,
    num_classes,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_layers,
    transformer_units,
    mlp_head_units,
):
    inputs = layers.Input(shape=input_shape)
    tokens = PatchTokenization(
        input_shape, patch_size, num_patches, projection_dim
    )(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(tokens)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(
        representation, hidden_units=mlp_head_units, dropout_rate=0.5
    )
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
