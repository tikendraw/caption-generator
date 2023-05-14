from dataclasses import dataclass
from pathlib import Path


@dataclass
class Configure():
    caption_file:Path
    image_dir:Path
    raw_caption_file:Path
    BATCH_SIZE:int

    IMG_SIZE : int
    CHANNELS : int

    IMG_SHAPE :tuple
    MAX_LEN : int
    EPOCHS  : int
    LEARNING_RATE : float
    UNITS : int
    glove_path : Path

    TEST_SIZE : float
    VAL_SIZE : float
    EMBEDDING_DIMENSION : int



config = Configure(
                    BATCH_SIZE = 8,
                    IMG_SIZE = 256,
                    CHANNELS = 3,
                    IMG_SHAPE = (256, 256, 3),
                    MAX_LEN = 50,
                    EPOCHS = 10,
                    LEARNING_RATE = 1e-2,
                    UNITS = 16,
                    raw_caption_file = Path('input/flickr30k/results.csv'),
                    caption_file = Path('input/flickr30k/results_cleaned.csv'),
                    image_dir = Path('./input/flickr30k/images'),
                    glove_path = Path("./embedding/glove.6B.50d.zip"),
                    TEST_SIZE = 0.05,
                    VAL_SIZE= 0.05,
                    EMBEDDING_DIMENSION = 50 # Do not change EMBEDDING_DIMENSION (glove.6B.50d)
                    )

