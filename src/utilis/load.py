
import logging
import pandas as pd
import numpy as np
from PIL import Image

def load_csv():
    styles_metadata = pd.read_csv("data/raw/styles.csv", delimiter=';').set_index("id")
    images_metadata = pd.read_csv("data/raw/images.csv", delimiter=';')
    image_ids = np.sort(styles_metadata.index)
    image_address = ["data/raw/images/"+str(id)+".jpg" for id in image_ids]
    image_address_dictionary = {int(k):v for k,v in zip(image_ids,image_address )}
    logging.info(f'{datetime.now()}: metadata csv files are successfully loaded')

    return styles_metadata, images_metadata, image_address_dictionary

def load_images(image_address_dictionary):
    image_set = {}
    missed_items_ids = []
    for id in image_address_dictionary:
        address = image_address_dictionary[id]
        try:
            image_set[id] = Image.open(address)
        except:
            missed_items_ids.append(id)
    logging.info(f'{datetime.now()}: These images are missed: {missed_items_ids}')

    return image_set