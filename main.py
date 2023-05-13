

import os
if 'google.colab' in str(get_ipython()):
    os.system('git clone https://github.com/tikendraw/caption-generator.git -q')
    os.chdir('caption-generator')

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

from preprocessing import clean_words, clean_df
from config import config

seed_value = 12321
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


IMG_SIZE           = config.IMG_SIZE
CHANNELS           = config.CHANNELS
BATCH_SIZE         = config.BATCH_SIZE
EPOCHS             = config.EPOCHS
IMG_SHAPE          = config.IMG_SHAPE
MAX_LEN            = config.MAX_LEN
UNITS              = config.UNITS
LAEARNING_RATE     = config.LEARNING_RATE
image_dir          = config.image_dir
caption_file       = config.caption_file
VOCAB_SIZE         = tokenizer.vocabulary_size()
# if __name__=='__main__':

df = pd.read_csv(caption_file)


tokenizer = TextVectorization(standardize=preprocess_text)


tokenizer.adapt(df['comment'])


word_to_id = tf.keras.layers.StringLookup(
    vocabulary=tokenizer.get_vocabulary(), mask_token='', oov_token='[UNK]')
id_to_word = tf.keras.layers.StringLookup(vocabulary=tokenizer.get_vocabulary(
), mask_token='', oov_token='[UNK]', invert=True)



glove_api_command = 'kaggle datasets download -d watts2/glove6b50dtxt'
glove_url = 'https://www.kaggle.com/datasets/watts2/glove6b50dtxt'
if 'google.colab' in sys.modules:

    download_kaggle_dataset(glove_api_command)
    os.makedirs('embedding', exist_ok = True)
    shutil.move('glove6b50dtxt.zip', 'embedding/glove.6B.50d.zip',)
glove_path = Path("./embedding/glove.6B.50d.zip")



np.save("./embedding/embedding_matrix.npy", embedding_matrix, allow_pickle=True)

ZipFile("embedding_matrix.zip", mode="w").write(
    "./embedding/embedding_matrix.npy"
)


resnet = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_tensor=tf.keras.layers.Input(shape=IMG_SHAPE)

)

resnet.trainable = False











TEST_SIZE = .05
VAL_SIZE = .05

train, val = train_test_split(
    df[['image_path', 'comment']],  test_size=VAL_SIZE, random_state=11)
train, test = train_test_split(
    train[['image_path', 'comment']],  test_size=TEST_SIZE, random_state=11)



train_data = tf.data.Dataset.from_tensor_slices(
    (train.image_path, train.comment))
test_data = tf.data.Dataset.from_tensor_slices((test.image_path, test.comment))
val_data = tf.data.Dataset.from_tensor_slices((val.image_path, val.comment))





train_data = train_data.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(mapper).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
val_data = val_data.map(mapper).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)





word_dict = {word: i for i, word in enumerate(textvectorizer.get_vocabulary())}


embedding_matrix = embedding_matrix_creater(
    50, word_index=word_dict
)





model, img_model = get_model()
print(model.summary(), img_model.summary())




model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=masked_loss,
              metrics=[masked_acc, masked_loss])

os.makedirs('log', exist_ok=True)
csv_logger = CSVLogger('./log/training.log')

EPOCHS = 10
steps_per_epoch = int(0.1*(len(train_data) / EPOCHS))
validation_steps =  int(.2*(len(val_data) / EPOCHS))


history = model.fit(train_data,
                    epochs=EPOCHS,
                    validation_data=val_data,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    callbacks=[
                        csv_logger, create_model_checkpoint(model_name = 'capgen', save_dir = './drive/MyDrive/caption_generator', monitor = 'masked_acc')
                                ]
                    )

model.save(f'/content/drive/MyDrive/caption_generator/{datetime.datetime.now()}-{EPOCHS}.h5')




start_token = word_to_id('startseq') 
end_token = word_to_id('endseq') 



aa = generate_caption(random_image_path, model, tokenizer)
print(aa)
