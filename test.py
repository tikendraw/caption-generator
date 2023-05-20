from transformer import CaptionGenerator
import os
import sys

import os
import sys
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
import regex as re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from sklearn.model_selection import train_test_split


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
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, TensorBoard
from model import LearningRateDecayCallback, get_model, masked_acc, masked_loss
from preprocessing import preprocess_text, embedding_matrix_creater, mapper, clean_words, clean_df
from utils import create_model_checkpoint

from config import config

from get_data import download_dataset
from funcyou.dataset import download_kaggle_dataset

from funcyou.utils import printt, dir_walk
import matplotlib.pyplot as plt
from nltk import word_tokenize
import nltk
from collections import Counter
import regex as re
import yaml


config_file_path = './config.yaml'

# Read the config file 
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




DFF = 1024
VOCAB_SIZE = 7000

capget = CaptionGenerator(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, VOCAB_SIZE, PATCH_SIZE, NUM_PATCHES, dropout_rate=0.1)
print(capget.summary())