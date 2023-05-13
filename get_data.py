import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocessing
from tensorflow.keras.layers import GlobalAveragePooling2D
from funcyou.dataset import download_kaggle_dataset
from pathlib import Path
from config import config
from preprocessing import clean_df

dataset_url = Path('https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset')
api_command = "kaggle datasets download -d hsankesara/flickr-image-dataset"

dest_caption_file_dir = config.caption_file
src_caption_file_dir = Path("./flickr30k_images/flickr30k_images/results.csv")

def move_to_dest():
    src_dir = "./flickr30k_images/flickr30k_images/flickr30k_images/"
    dst_dir = config.image_dir
    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob(f"{src_dir}*")

    for file in files:
        filename = file.split("/")[-1]
        dst_path = f"{dst_dir}/{filename}"
        shutil.copy(file, dst_path)


    shutil.copy(src_caption_file_dir, dest_caption_file_dir)



if __name__=='__main__':

    # print("Downloading...")
    # download_kaggle_dataset(api_command, kaggle_json_filepath='./kaggle.json')
    # print("Downloaded.")
    

    # print('Extracting...')
    # with ZipFile('flickr-image-dataset.zip') as zip:
    #     zip.extractall()

    # print('Moving...')
    # move_to_dest()
    # print('Done')

    clean_df()
    print('Cleaned.')
