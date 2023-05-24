

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




def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)        

    return tf.cast(pos_encoding, dtype=tf.float32)


# Positional embedding For Image
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x



class Patches(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()


    def call(self, images):
        batch_size = tf.shape(images)[0]
        
        image_features = tf.keras.applications.resnet.preprocess_input(images)
        image_features = resnet(image_features, training=False)
        
        # image_features = GlobalAveragePooling2D()(image_features)
        image_features = tf.squeeze(image_features)
        image_features = tf.reshape(image_features, (batch_size,image_features.shape[-3]*image_features.shape[-2], -1))

        return image_features


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_patches=64):
        super().__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=d_model)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=d_model
        )

    def call(self, patch):
        
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        tf.print(positions.shape)
        return self.projection(patch) + self.position_embedding(positions)

# Attention
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_attn_scores=None
    
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    # tf.print('attn_output: ',attn_output.shape)
    x = self.add([x, attn_output])
    # tf.print('concat: ',x.shape)

    x = self.layernorm(x)
    # tf.print('layernorm: ',x.shape)

    return x

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x


class FeedForword(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate = 0.1):
        super().__init__()

        
        self.seq = tf.keras.Sequential([
            Dense(dff, activation = 'relu'),
            Dense(d_model),
            Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
        
    def call(self, x):
        x = self.add([x, self.seq(x)])
        return self.layernorm(x)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        
        self.self_attention = GlobalSelfAttention(
            key_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate
        )
        
        self.ffn = FeedForword(d_model=d_model, dff=dff,dropout_rate=dropout_rate)
        

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
        


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, patch_size, num_patches, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.dropout_rate = dropout_rate
        
        
        self.patches = Patches()
        # Encode patches.
        self.encoded_patches = PatchEncoder(num_patches=num_patches, d_model=d_model)
        

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.patches(x)  # Shape `(batch_size, 64, 2048)`. reshaped outpt of resnet(8,8, 2048) with batch_size
        x = self.encoded_patches(x)
        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim = d_model, 
            dropout= dropout_rate
            )
        
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim = d_model, 
            dropout= dropout_rate
            )
        
        self.ffn = FeedForword(d_model=d_model, dff=dff,dropout_rate=dropout_rate)
               
        self.last_attn_scores = self.cross_attention.last_attn_scores

       
    def call(self, x, context):
        x = self.causal_attention(x)
        x = self.cross_attention(x=x, context = context)
        x = self.ffn(x)
        return x
        

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        
        self.num_layers = num_layers 
        self.d_model = d_model
        self.num_heads = num_heads 
        self.dff = dff
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate=0.1
        
        self.positional_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                        dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.last_attn_scores = None


    def call(self, x, context):
        # tf.print('x: ', x.shape)
        # tf.print('context: ', context.shape)
        
        x = self.positional_embedding(x)
        # tf.print('pos-emb x: ', x.shape)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x=x, context=context)
        
            
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        # tf.print('afte tra x : ', x.shape)

        return x


class CaptionGenerator(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, patch_size, num_patches, dropout_rate=0.1):
        super().__init__()

        self.encoder = Encoder(
                            num_layers=num_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            patch_size=patch_size,
                            num_patches=num_patches,
                            dropout_rate=dropout_rate,
                            )
        
        self.decoder = Decoder(
                            num_layers=num_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            vocab_size=vocab_size,
                            dropout_rate=dropout_rate,
                            )
        
        self.final_layer = tf.keras.layers.Dense(vocab_size
                               
        )        
        self.decoder = Decoder(
                            num_heads=num_heads,
                            num_layers=num_layers,
                            d_model=d_model,
                            dff=dff,
                            vocab_size=vocab_size,
                            dropout_rate=dropout_rate,
                            )
        
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        
    def call(self, inputs):  # sourcery skip: inline-immediately-returned-variable, use-contextlib-suppress
        img, txt  = inputs

        img = self.encoder(img)  # (batch_size, context_len, d_model)

        x = self.decoder(x=txt, context=img)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, max_len, target_vocab_size)
        
        try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits


if __name__=='__main__':
    print('transformer.py')