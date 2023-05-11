# Dependencies
import numpy as np
import pandas as pd
import os, random, math
import tensorflow as tf
import glob
import shutil
from zipfile import ZipFile
import datetime
import sys


# ! git clone https://github.com/tikendraw/funcyou -q


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tqdm import tqdm
from pathlib import Path


from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocessing
from tensorflow.keras.layers import (
    TextVectorization, Embedding, LSTM, GRU, Bidirectional, TimeDistributed, Dense, Attention, MultiHeadAttention, Flatten, Dropout,
    Concatenate, Activation, GlobalAveragePooling2D
    )
from tensorflow import keras
from tensorflow.keras import Input, layers
from tensorflow.keras.utils import array_to_img, img_to_array
import string
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

UNITS = 64
max_len = 50
VOCAB_SIZE = 20_000
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 5
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

resnet = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_tensor=tf.keras.layers.Input(shape=(256, 256, 3))

)

def get_model():

    embdding_layer = Embedding(input_dim = VOCAB_SIZE, output_dim = UNITS, input_length = max_len, mask_zero = True)
    rnn = LSTM(UNITS, return_sequences=True, return_state=True)

    # image inputs
    image_input = Input(shape=IMG_SHAPE)
    print(image_input.shape)
    
    x = resnet_preprocessing(image_input)
    print('preprocess: ', x.shape)
    
    x = resnet(x)
    print('resnet: ',x.shape)

    x = GlobalAveragePooling2D()(x)  # add a pooling layer\
    print('polling: ',x.shape)

    x = Dense(UNITS*max_len)(x)
    print('dense: ',x.shape)

    x = tf.reshape(x, (-1, max_len, UNITS))
    print('reshape: ',x.shape)
    print('')

    # text inputs
    text_input = Input(shape=(max_len,))
    print('text_input: ',text_input.shape)

    i = embdding_layer(text_input)
    print('embedding: ',i.shape)


    i, j, k = rnn(i)
    i, _, _ = rnn(i, initial_state=[j,k])
    print('i:', i.shape)

    #  attention between x and i
    l = Attention()([x, i])
    ll = Attention()([i, x])
    print('attentions: ',l.shape, ll.shape)
    
    #  concatnate x and i
    # m = Concatenate()([x, i, l, ll ])
    m = Concatenate()([x, i ])
    mm = Concatenate()([l, ll ])
    print('concat attention: ',m.shape, mm.shape)

    mm = Attention()([m,mm])
    print('mm attentions: ',mm.shape)

    m = layers.Dot(axes=-1)([m,mm])
    print('dot m: ', m.shape)

    m = Dense(UNITS**2)(m)
    m = Dense(UNITS**2)(m)

    m = Dense(VOCAB_SIZE)(m)
    print('dense out: ',m.shape)

    m = Activation('softmax')(m)
    
    return keras.Model(inputs = [image_input, text_input], outputs = m)

get_model()

