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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

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



from funcyou.utils import dir_walk
from funcyou.dataset import download_kaggle_dataset
from config import config
import yaml

config_file_path = './config.yaml'

# Read the config file 
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)


BATCH_SIZE =        config['batch_size'] 
IMG_SIZE =          config['img_size'] 
CHANNELS =          config['channels'] 
IMG_SHAPE =         config['img_shape'] 
MAX_LEN =           config['max_len'] 
EPOCHS =            config['epochs'] 
LEARNING_RATE =     config['learning_rate'] 
UNITS =             config['units'] 
raw_caption_file =  config['raw_caption_file'] 
caption_file =      config['caption_file'] 
image_dir =         config['image_dir'] 
glove_path =        config['glove_path'] 
TEST_SIZE =         config['test_size'] 
VAL_SIZE=           config['val_size']
EMBEDDING_DIMENSION =   config['embedding_dimension'] 


def clean_words(x, words_to_keep):
    words = re.split(r'\W+', x)
    return ' '.join(w for w in words if w.lower() in words_to_keep)

def preprocess_text(text):

    text = tf.strings.lower(text)

    text = tf.strings.regex_replace(text, r'\d', '')

    # Remove any punctuations
    text = tf.strings.regex_replace(text, '[%s]' % re.escape(
        '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'), '')

    # Remove single characters
    text = tf.strings.regex_replace(text, r'\b\w\b', '')
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿|]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    # text = tf.strings.join([START_TOKEN, text, END_TOKEN], separator=' ')
    return text

def mapper(x, y, tokenizer):
    x = load_images_now(x)
    y = tokenizer(y)

    y_in = y[:-1]
    y_in =  tf.pad(y_in, [[0, MAX_LEN - tf.shape(y_in)[0]]] , constant_values=0)

    y_out = y[1:]
    y_out =  tf.pad(y_out, [[0, MAX_LEN - tf.shape(y_out)[0]]], constant_values=0)

    return (x, y_in), y_out

# load image model
resnet = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_tensor=tf.keras.layers.Input(shape=IMG_SHAPE))

resnet.trainable = False


@tf.function
def load_images_now(x):
    image_data = tf.io.read_file(x)
    image_features = tf.image.decode_jpeg(image_data, channels=CHANNELS)
    image_features = tf.image.resize(
        image_features, (IMG_SIZE, IMG_SIZE))
    image_features = tf.keras.applications.resnet.preprocess_input(
        image_features)
    image_features = tf.reshape(
        image_features, (1, IMG_SIZE, IMG_SIZE, CHANNELS))
    image_features = resnet(image_features)
    image_features = GlobalAveragePooling2D()(image_features)
    image_features = tf.squeeze(image_features)

    return image_features



def glove_embedding(path:Path) -> dict:
    with ZipFile(path) as f:
        f.extractall("./embedding/")
    

    embeddings_index = {}
    with open(path.parent / "glove.6B.50d.txt", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        f.close()

    os.remove("./embedding/glove.6B.50d.txt")
    return embeddings_index


def embedding_matrix_creater(EMBEDDING_DIMS, word_index):
    embeddings_index = glove_embedding(glove_path)
    embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIMS))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
    return embedding_matrix


def tokens_to_text(tokens):
    words = id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, f'^ *{START_TOKEN} *', '')
    result = tf.strings.regex_replace(result, f' *{END_TOKEN} *$', '')
    return result

from config import config
from utils import word_count_df







if __name__=='__main__':
    print('preprocessing.py')