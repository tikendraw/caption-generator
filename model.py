import numpy as np
import pandas as pd
import random, math
import tensorflow as tf
import glob
import shutil
from zipfile import ZipFile
import datetime
import sys
from functools import cache
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import pad_sequences
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from collections import Counter
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocessing
from tensorflow.keras.layers import (
    TextVectorization, Embedding, LSTM, GRU, Bidirectional, TimeDistributed, Dense, Attention, MultiHeadAttention, Flatten, Dropout,
    Concatenate, Activation, GlobalAveragePooling2D
    )
from tensorflow.keras.layers import LSTM, Embedding, Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow import keras
from tensorflow.keras.utils import array_to_img, img_to_array
import string
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
import regex as re
from config import config



BATCH_SIZE = config.BATCH_SIZE
IMG_SIZE = config.IMG_SIZE
CHANNELS = config.CHANNELS
IMG_SHAPE = config.IMG_SHAPE
MAX_LEN = config.MAX_LEN
UNITS = config.UNITS
EPOCHS = config.EPOCHS
EMBEDDING_DIMENSION =   config.EMBEDDING_DIMENSION 


def get_model(embedding_matrix, VOCAB_SIZE):
    encoder = LSTM(UNITS, return_sequences=True, return_state=True)
    decoder = LSTM(UNITS, return_sequences=True)

    embedding = Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIMENSION,
        mask_zero=True,
        input_length=MAX_LEN,
        trainable=False,
    )
    embedding.build([None])
    embedding.set_weights([embedding_matrix])


    image_input = Input(shape=(2048,), name = 'image_input')
    print('image_input: ',image_input.shape)

    x = Dropout(.3)(image_input)
    print('enc dropout: ',x.shape)
    
    x = Dense(MAX_LEN*UNITS, activation = 'relu', kernel_initializer = 'glorot_uniform' )(x)
    print('Dense: ',x.shape)
    
    x = tf.reshape(x, (-1, MAX_LEN, UNITS))
    print('reshape: ',x.shape)


    txt_input = Input(shape=(MAX_LEN,), name = 'text_input')
    print('txt_input: ',txt_input.shape)
    
    i = embedding(txt_input)
    print('text_embedding: ',i.shape)
    
    i, j, k = encoder(i)
    print('encoder output: ',i.shape)
    
    i = Dropout(.3)(i)
    print('enc dropout: ',i.shape)
    
    i = decoder(i, initial_state=[j, k])
    print('decoder output: ',i.shape)
    
    i = Dropout(.3)(i)
    print('decoder dropout: ',i.shape)

    l = Attention()([x, i])
    ll = Attention()([i, x])
    print('attention: ',l.shape, ll.shape)

    m = Concatenate()([x, i, l, ll])
    print('concat: ', m.shape)

    m = Dropout(.3)(m)
    print('concat dropout: ',m.shape)

    m = Dense(MAX_LEN*UNITS)(m)
    print('Dense1: ', m.shape)
    
    m = Dense(MAX_LEN*UNITS)(m)
    print('Dense2: ', m.shape)

    m = Dense(VOCAB_SIZE)(m)
    print('Dense3_final: ', m.shape)
    return Model(inputs=[image_input, txt_input], outputs=m)




def masked_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction='none')
    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    matchh = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(matchh)/tf.reduce_sum(mask)



class LearningRateDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_learning_rate, decay_rate, decay_steps):
        super(LearningRateDecayCallback, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0 and epoch % self.decay_steps == 0:
            updated_lr = self.initial_learning_rate * (self.decay_rate ** (epoch // self.decay_steps))
            tf.keras.backend.set_value(self.model.optimizer.lr, updated_lr)
            print(f"Learning rate updated to: {updated_lr}")




if __name__=='__main__':
    print('model.py')

