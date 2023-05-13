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




config = Configure(
                    BATCH_SIZE = 32,
                    IMG_SIZE = 256,
                    CHANNELS = 3,
                    IMG_SHAPE = (256, 256, 3),
                    MAX_LEN = 50,
                    EPOCHS = 10,
                    LEARNING_RATE = 1e-3,
                    UNITS = 16,
                    raw_caption_file = Path('input/flickr30k/results.csv'),
                    caption_file = Path('input/flickr30k/results_cleaned.csv'),
                    image_dir = Path('./input/flickr30k/images')
                    )


print(config.UNITS)