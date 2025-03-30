
import logging
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

# load csv files with metadata
def load_csv():
    styles_metadata = pd.read_csv("data/raw/styles.csv", delimiter=';').set_index("id")
    images_metadata = pd.read_csv("data/raw/images.csv", delimiter=';')
    # store database images' ids
    image_ids = np.sort(styles_metadata.index)
    # create a list with full path to every dataase image
    image_address = ["data/raw/images/"+str(id)+".jpg" for id in image_ids]
    # create dictionary with images' ids and paths to these images
    image_address_dictionary = {int(k):v for k,v in zip(image_ids,image_address )}
    logging.info(f'{datetime.now()}: metadata csv files are successfully loaded')

    return styles_metadata, images_metadata, image_address_dictionary

# load database images
def load_images(image_address_dictionary):
    # dictinoary to store images (image_id:image)
    image_set = {}
    missed_items_ids = [] # list to store images that cannot be found
    # read images using their paths and add tot he dictionaet
    for id in image_address_dictionary:
        address = image_address_dictionary[id]
        try:
            image_set[id] = Image.open(address)
        except:
            missed_items_ids.append(id)
    logging.info(f'{datetime.now()}: These images are missed: {missed_items_ids}')

    # reutn images in dictionary, with image_id as key
    return image_set