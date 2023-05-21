
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

if not os.path.exists('funcyou'):
	os.system('git clone https://github.com/tikendraw/funcyou -q')
os.system('pip install funcyou/. -q')

from funcyou.utils import dir_walk
from funcyou.dataset import download_kaggle_dataset

import nltk
from nltk import word_tokenize

# Visualizing
def plot_image_with_captions(df, n: int = 3):
    for _ in range(n):
        sample_image_name = df['image_name'].sample(1).values[0]

        # Reading image
        sample_path = os.path.join(image_dir, sample_image_name)
        fig = plt.figure(figsize=(25, 5))
        ax = fig.add_subplot(1, 2, 1)
        image = plt.imread(sample_path)
        plt.imshow(image)

        # captions
        all_captions = df[df['image_name'] == sample_image_name]['comment']
        ax = fig.add_subplot(1, 2, 2)
        plt.axis('off')
        for num, caption in enumerate(all_captions.values):
            caption = f'{num+1} : {caption}'
            ax.text(0, 0.85 - num*(1/8), caption, horizontalalignment='left', verticalalignment='bottom',
                    multialignment='left', fontsize='x-large', transform=ax.transAxes)


def word_count_df(x:pd.Series):
    text_data = x.str.lower().str.cat(sep=' ')
    words = word_tokenize(text_data) 
    word_count = Counter(words)
    unique_words = set(words)
    len(word_count.keys()), len(word_count.values())

    countdf = pd.DataFrame([word_count.keys(), word_count.values()]).T
    countdf.columns = ['word', 'counts']
    return countdf


def create_model_checkpoint(model_name, save_dir, monitor: str = 'val_loss', verbose: int = 0, save_best_only: bool = True, save_weights_only: bool = False,
                            mode: str = 'auto', save_freq='epoch', options=None, initial_value_threshold=None, **kwargs):
    model_name = f'{model_name}-{str(datetime.datetime.now())}'
    dirs = os.path.join(save_dir, model_name)

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    return tf.keras.callbacks.ModelCheckpoint(
        dirs,
        monitor=monitor,
        verbose=verbose,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        mode=mode,
        save_freq=save_freq,
        options=options,
        initial_value_threshold=initial_value_threshold,
        **kwargs)




def create_initial_input(MAX_LEN):

    start_token = word_to_id('startseq')

    initial_input = [0] * MAX_LEN

    initial_input[0] = start_token.numpy()

    initial_input = tf.reshape(initial_input,(MAX_LEN,))
    return initial_input

def generate_caption(image_path, model, tokenizer, beam_size=5):
    ii = plt.imread(image_path)
    plt.imshow(ii)

    features = load_images_now(image_path)
    features = tf.reshape(features, (1, features.shape[0]))

    beams = [[start_token]] * beam_size

    for _ in range(MAX_LEN):
        for i, beam in enumerate(beams):
            sequence = tokenizer([beam.decode('utf-8')])
            sequence = tf.pad(sequence, [[0, 0], [0, MAX_LEN - tf.shape(sequence)[1]]])

            logits = model.predict([features, sequence], verbose = 0)
            next_word_idx = tf.argmax(logits, axis=-1)

            next_word = id_to_word(next_word_idx.numpy()[0])

            beams[i].append(next_word)

            beams.sort(key=lambda x: x[-1], reverse=True)

            beams = beams[:beam_size]

        caption = beams[0]
    return ' '.join(caption[1:])


if __name__=='__main__':
    print('utils.py')