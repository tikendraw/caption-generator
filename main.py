# Dependencies

import os
import sys
if 'google.colab' in sys.modules:
    os.system('git clone https://github.com/tikendraw/caption-generator.git -q')
    os.chdir('caption-generator')


if not os.path.exists('funcyou'):
	os.system('git clone https://github.com/tikendraw/funcyou -q')

# os.system('pip install funcyou/. -q')


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

seed_value = 12321
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


BATCH_SIZE =    config.BATCH_SIZE
IMG_SIZE =      config.IMG_SIZE
CHANNELS =      config.CHANNELS
IMG_SHAPE =     config.IMG_SHAPE
MAX_LEN =       config.MAX_LEN
EPOCHS =        config.EPOCHS
LEARNING_RATE = config.LEARNING_RATE
UNITS =         config.UNITS
raw_caption_file =  config.raw_caption_file
caption_file =  config.caption_file
image_dir =     config.image_dir
glove_path =    config.glove_path
TEST_SIZE =     config.TEST_SIZE
VAL_SIZE=       config.VAL_SIZE
EMBEDDING_DIMENSION =   config.EMBEDDING_DIMENSION 

pathh = '/content/kaggle.json'
if 'google.colab' in sys.modules:
    download_dataset(pathh)

# Reading
df = pd.read_csv(caption_file)
print(df.info())

df = df[df.comment_number == 1]
print(df.shape)
# Tokenize



#tokenizer
tokenizer = TextVectorization(standardize=preprocess_text)
tokenizer.adapt(df['comment'])


word_to_id = tf.keras.layers.StringLookup(vocabulary=tokenizer.get_vocabulary(), mask_token='', oov_token='[UNK]')
id_to_word = tf.keras.layers.StringLookup(vocabulary=tokenizer.get_vocabulary(), mask_token='', oov_token='[UNK]', invert=True)




#GLOVE embedding
glove_api_command = 'kaggle datasets download -d watts2/glove6b50dtxt'
glove_url = 'https://www.kaggle.com/datasets/watts2/glove6b50dtxt'

# if 'google.colab' in sys.modules:

#     download_kaggle_dataset(glove_api_command)
#     os.makedirs('embedding', exist_ok = True)
#     shutil.move('glove6b50dtxt.zip', 'embedding/glove.6B.50d.zip',)


# creating embedding matrix
word_dict = {word: i for i, word in enumerate(tokenizer.get_vocabulary())}

# Creating embedding matrix
embedding_matrix = embedding_matrix_creater(EMBEDDING_DIMENSION, word_index=word_dict)

# # Saving embedding_matrix for further use
# np.save("./embedding/embedding_matrix.npy", embedding_matrix, allow_pickle=True)
# # compressing
# ZipFile("embedding_matrix.zip", mode="w").write(
#     "./embedding/embedding_matrix.npy")


# load image model
resnet = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_tensor=tf.keras.layers.Input(shape=IMG_SHAPE))

resnet.trainable = False


# Creating dataset
TEST_SIZE = config.TEST_SIZE
VAL_SIZE =  config.VAL_SIZE

train, val = train_test_split(
    df[['image_path', 'comment']],  test_size=VAL_SIZE, random_state=11)
train, test = train_test_split(
    train[['image_path', 'comment']],  test_size=TEST_SIZE, random_state=11)



train_data = tf.data.Dataset.from_tensor_slices((train.image_path, train.comment))
test_data = tf.data.Dataset.from_tensor_slices((test.image_path, test.comment))
val_data = tf.data.Dataset.from_tensor_slices((val.image_path, val.comment))


train_data = train_data.map(lambda x,y:mapper(x, y, tokenizer)).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
test_data =   test_data.map(lambda x,y:mapper(x, y, tokenizer)).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
val_data =     val_data.map(lambda x,y:mapper(x, y, tokenizer)).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

# resnet_output_flattened_shape = 8*8*2048

print("Number of training samples: %d" %
      tf.data.experimental.cardinality(train_data))
print("Number of validation samples: %d" %
      tf.data.experimental.cardinality(val_data))
print("Number of test samples: %d" %
      tf.data.experimental.cardinality(test_data))

VOCAB_SIZE = tokenizer.vocabulary_size()
print("Vocabulary size: %d" % VOCAB_SIZE)


model = get_model(embedding_matrix, VOCAB_SIZE)
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=masked_loss,
              metrics=[masked_acc, masked_loss])


# model = tf.keras.models.load_model('/content/drive/MyDrive/cap-gen/2023-05-14 08:04:36.705364-30.tf')

# Example usage
initial_lr = LEARNING_RATE
decay_rate = 0.001
decay_steps = 10

decay_callback = LearningRateDecayCallback(initial_lr, decay_rate, decay_steps)

os.makedirs('log', exist_ok=True)
csv_logger = CSVLogger('./log/training.log')
tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
print(initial_lr)
EPOCHS = 3

print(len(train_data) // EPOCHS, len(val_data) // EPOCHS)

steps_per_epoch = int(0.1*(len(train_data) / EPOCHS))
validation_steps =  int(.2*(len(val_data) / EPOCHS))
print(steps_per_epoch, validation_steps)


history = model.fit(train_data,
                    epochs=EPOCHS,
                    validation_data=val_data,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    callbacks=[
                        # decay_callback,
                        csv_logger, create_model_checkpoint(model_name = 'capgen', save_dir = 'checkpoints', monitor = 'masked_acc')
                                ]
                    )

from funcyou.plot import plot_history
import pandas as pd


plot_history(history, plot = ['masked_loss','masked_acc'], split = ['train','val'], epoch= EPOCHS, figsize = (20,10), colors = ['r','b'])
from google.colab import drive
drive.mount('/content/drive')

# model.save(f'/content/drive/MyDrive/cap-gen/{datetime.datetime.now()}-{EPOCHS}.tf')

pred = model.predict(test_data.take(1))
# print((pred.shape))

# start_token_id = word_to_id('startseq') 
# end_token_id = word_to_id('endseq') 



# aa = generate_caption(random_image_path, model, tokenizer)
# print(aa)

# pred.shape
from preprocessing import tokens_to_text
tokens_to_text(pred, tokenizer)
