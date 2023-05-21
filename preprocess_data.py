
import os
import pandas as pd
import regex as re
from preprocessing import  clean_words
from utils import  word_count_df
import polars as pl
from collections import Counter

def check_words_in_list(string, words:list):
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




def clean_the_df(filepath, save_dir):
    
    df = pl.read_csv(filepath, sep='|', ignore_errors = True)
    df.columns = [str(i).strip().lower() for i in df.columns]
    
    df = df.drop_nulls()

    # Convert 'comment_number' column to int
    df = df.with_columns([
        pl.col('comment_number').cast(pl.Int64, strict=False).alias('comment_number'),        
        ])

    # Remove rows with null values
    df = df.drop_nulls()


    words = set(df['comment'])

    word_list = []
    for sentence in words:
        sentence = sentence.lower()
        words = sentence.split()
        word_list.extend(words)

    count_dict = Counter(word_list)

    countdf = pd.DataFrame([count_dict.keys(), count_dict.values()]).T
    countdf.columns = ['word', 'counts']
    countdf.to_csv('word_count.csv')

    words_to_keep = set(countdf[countdf.counts>4]['word'].values)

    # Add start and end tokens to 'comment' column
    START_TOKEN = 'startseq '
    END_TOKEN = ' endseq'

    # Clean words in 'comment' column
    df = df.with_columns([
            pl.col("comment").apply(lambda x: re.sub(r'[.?!,¿|]', r' \g<0> ', x)).alias('comment')             
            ])
    
    # Clean words in 'comment' column
    df = df.with_columns([
            pl.col("comment").apply(lambda x: clean_words(x, words_to_keep=words_to_keep)).alias('comment')
            ])
    
    # Clean words in 'comment' column
    df = df.with_columns([
            pl.col("comment").apply(lambda x: set(x.lower().split()).issubset(words_to_keep)).alias('no_rare_words')
            ])
    
    # Clean words in 'comment' column
    df = df.with_columns([
            pl.col("comment").apply(lambda x: START_TOKEN+ ' ' + x + ' '+END_TOKEN ).alias('comment')             
        ])

    df = df.with_columns([
            pl.col("comment").apply(lambda x: len(x.split())).alias('sent_len')
        ])


    # Save the dataframe
    if save_dir.endswith('.csv'):
        df.write_csv(save_dir)
    else:
        df.write_csv(os.path.join(save_dir, 'cleaned.csv'))