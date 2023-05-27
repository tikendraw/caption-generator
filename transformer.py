

import os, sys
import numpy as np
import pandas as pd
import random, math
import tensorflow as tf
import datetime
from pathlib import Path
import regex as re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


from tensorflow.keras.layers import (
    TextVectorization, Embedding, Dense, Attention, MultiHeadAttention, Flatten, Dropout,
    Concatenate, Activation, GlobalAveragePooling2D, Input
    )

import string
from tensorflow.keras import layers
from tensorflow import keras 

from config import config
import yaml

# Define the path to your config file
config_file_path = './config.yaml'

# Read the config file and load its content into a Python object
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)




RAW_CAPTION_FILE                        = config['raw_caption_file']
CAPTION_FILE                            = config['caption_file']
IMAGE_DIR                               = config['image_dir']
IMG_SIZE                                = config['img_size']
CHANNELS                                = config['channels']
IMG_SHAPE                               = config['img_shape']
MAX_LEN                                 = config['max_len']
BATCH_SIZE                              = config['batch_size']
EPOCHS                                  = config['epochs']
LEARNING_RATE                           = config['learning_rate']
UNITS                                   = config['units']
TEST_SIZE                               = config['test_size']
VALIDATION_SIZE                         = config['val_size']
EMBEDDING_DIMENSION                     = config['embedding_dimension']
GLOVE_PATH                              = config['glove_path']
D_MODEL                                 = config['d_model']
NUM_HEADS                               = config['num_heads']
NUM_LAYERS                               = config['num_layers']    
    
PATCH_SIZE                              = config['patch_size']    
TRANSFORMER_LAYERS                      = config['transformer_layers']            

NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2 
# trick here is to match max_len to num_patches for matching the shapes for concatination


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)


# def positional_encoding(length, depth):
#     depth = depth/2

#     positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
#     depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

#     angle_rates = 1 / (10000**depths)         # (1, depth)
#     angle_rads = positions * angle_rates      # (pos, depth)

#     pos_encoding = np.concatenate(
#         [np.sin(angle_rads), np.cos(angle_rads)],
#         axis=-1)        

#     return tf.cast(pos_encoding, dtype=tf.float32)


# # Positional embedding For Image
# class PositionalEmbedding(tf.keras.layers.Layer):
#   def __init__(self, vocab_size, d_model):
#     super().__init__()
#     self.d_model = d_model
#     self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
#     self.pos_encoding = positional_encoding(length=2048, depth=d_model)

#   def compute_mask(self, *args, **kwargs):
#     return self.embedding.compute_mask(*args, **kwargs)

#   def call(self, x):
#     length = tf.shape(x)[1]
#     x = self.embedding(x)
#     # This factor sets the relative scale of the embedding and positonal_encoding.
#     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#     x = x + self.pos_encoding[tf.newaxis, :length, :]
#     return x


# class Patches(tf.keras.layers.Layer):
#     def __init__(self, patch_size):
#         super().__init__()
#         self.patch_size = patch_size

#     def call(self, images):
#         batch_size = tf.shape(images)[0]
#         patches = tf.image.extract_patches(
#             images=images,
#             sizes=[1, self.patch_size, self.patch_size, 1],
#             strides=[1, self.patch_size, self.patch_size, 1],
#             rates=[1, 1, 1, 1],
#             padding="VALID",
#         )
#         patch_dims = patches.shape[-1]
#         # (patches.shape)
#         patches = tf.reshape(patches, [batch_size, -1, patch_dims])
#         return patches


# class PatchEncoder(tf.keras.layers.Layer):
#     def __init__(self, num_patches, d_model):
#         super().__init__()
#         self.num_patches = num_patches
#         self.projection = Dense(units=d_model)
#         self.position_embedding = Embedding(
#             input_dim=num_patches, output_dim=d_model
#         )

#     def call(self, patch):
        
#         positions = tf.range(start=0, limit=self.num_patches, delta=1)
# #         tf.print(positions.shape)
#         return self.projection(patch) + self.position_embedding(positions)

# Embedding

class SeqEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_length, d_model):
        super().__init__()
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_length, output_dim=d_model)

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            mask_zero=True)

        self.add = tf.keras.layers.Add()

    def call(self, seq):
        seq = self.token_embedding(seq) # (batch, seq, d_model)

        x = tf.range(tf.shape(seq)[1])  # (seq)
        x = x[tf.newaxis, :]  # (1, seq)
        x = self.pos_embedding(x)  # (1, seq, d_model)

        return self.add([seq,x])


# Attention

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add() 
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x, context, **kwargs):
        attn, attention_scores = self.mha(
                    query=x, value=context,
                    return_attention_scores=True)

        self.last_attention_scores = attention_scores

        x = self.add([x, attn])
        return self.layernorm(x)



class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        # Use Add instead of + so the keras mask propagates through.
        self.add = tf.keras.layers.Add() 
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attn = self.mha(query=x, value=x,
                        use_causal_mask=True)
        x = self.add([x, attn])
        return self.layernorm(x)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dff, d_model, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(units=dff, activation='relu'),
            tf.keras.layers.Dense(units=d_model),
            tf.keras.layers.Dropout(rate=dropout_rate),
        ])

        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = x + self.seq(x)
        return self.layernorm(x)



class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, dff, d_model, num_heads=1, dropout_rate=0.1):
        super().__init__()

        self.self_attention = CausalSelfAttention(num_heads=num_heads,
                                                    key_dim=d_model,
                                                    dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads,
                                                key_dim=d_model,
                                                dropout=dropout_rate)
        self.ff = FeedForward(dff=dff, d_model=d_model, dropout_rate=dropout_rate)


    def call(self, inputs, training=False):
        img_in, txt_in = inputs

        # Text input
        txt_in = self.self_attention(x=txt_in)

        txt_in = self.cross_attention(x = txt_in, context=img_in)

        self.last_attention_scores = self.cross_attention.last_attention_scores

        txt_in = self.ff(txt_in)

        return txt_in        


class TokenOutput(tf.keras.layers.Layer):
    def __init__(self, tokenizer, banned_tokens=('', '[UNK]', '[START]'), **kwargs):
        super().__init__()

        self.dense = tf.keras.layers.Dense(
            units=tokenizer.vocabulary_size(), **kwargs)
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens

        self.bias = None

    def adapt(self, ds):
        counts = collections.Counter()
        vocab_dict = {name: id 
                        for id, name in enumerate(self.tokenizer.get_vocabulary())}

        for tokens in tqdm.tqdm(ds):
            counts.update(tokens.numpy().flatten())

        counts_arr = np.zeros(shape=(self.tokenizer.vocabulary_size(),))
        counts_arr[np.array(list(counts.keys()), dtype=np.int32)] = list(counts.values())

        counts_arr = counts_arr[:]
        for token in self.banned_tokens:
            counts_arr[vocab_dict[token]] = 0

        total = counts_arr.sum()
        p = counts_arr/total
        p[counts_arr==0] = 1.0
        log_p = np.log(p)  # log(1) == 0

        entropy = -(log_p*p).sum()

        print()
        print(f"Uniform entropy: {np.log(self.tokenizer.vocabulary_size()):0.2f}")
        print(f"Marginal entropy: {entropy:0.2f}")

        self.bias = log_p
        self.bias[counts_arr==0] = -1e9

    def call(self, x):
        x = self.dense(x)
        # TODO(b/250038731): Fix this.
        # An Add layer doesn't work because of the different shapes.
        # This clears the mask, that's okay because it prevents keras from rescaling
        # the losses.
        return x + self.bias



if __name__=='__main__':
    print('transformer.py')
