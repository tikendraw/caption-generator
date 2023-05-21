
import os
import pandas as pd
import regex as re
from preprocessing import  clean_words
from utils import  word_count_df

from collections import Counter

def check_words_in_list(string, words):
    """
    Checks if any words in a string are not in a provided list.

    Args:
    string: The string to check.
    words: The list of words to check against.

    Returns:
    True if any words in the string are not in the provided list, False otherwise.
    """
    string = str(string).lower()
    words_in_string = set(string.split())
    words_in_list = set(words)

    return bool(words_in_string - words_in_list)




def clean_the_df(filepath, save_dir,IMAGE_DIR):
    # sourcery skip: use-fstring-for-concatenation
    df = pd.read_csv(filepath, sep='|')

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop row at index 19999
    df.drop(19999, inplace=True)  # Bad value at index 19999

    # Convert 'comment_number' column to numeric
    df['comment_number'] = pd.to_numeric(df['comment_number'])

    # Remove rows with null values
    df = df.dropna()

    # Count words and filter rare words
    countdf = word_count_df(df['comment'])
    words_to_keep = set(countdf.loc[countdf['counts'] > 4, 'word'])

    # Check for rare words in 'comment' column
    df['has_rare_words'] = df['comment'].map(lambda x: check_words_in_list(x, words_to_keep))
    # Clean words in 'comment' column
    df['comment'] = df['comment'].map(lambda x: clean_words(x, words_to_keep=words_to_keep))
    # Check for rare words again
    df['has_rare_words2'] = df['comment'].map(lambda x: check_words_in_list(x, words_to_keep))

    # Add start and end tokens to 'comment' column
    START_TOKEN = 'startseq'
    END_TOKEN = 'endseq'
    
    df['comment'] = START_TOKEN + ' ' + df['comment'] + ' ' + END_TOKEN
    # Create 'image_path' column
    df['image_path'] = os.path.join(IMAGE_DIR, df['image_name'])
    # Calculate sentence length
    df['sent_length'] = df['comment'].str.split().str.len()
    # Check if image file exists
    df['img_exists'] = df['image_path'].apply(os.path.isfile)

    # Save the dataframe
    if save_dir.endswith('.csv'):
        df.to_csv(save_dir, index=False)
    else:
        df.to_csv(os.path.join(save_dir, 'cleaned.csv'), index=False)
