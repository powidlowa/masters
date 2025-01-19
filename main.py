import logging
from src.helpers import load_csv

logging.basicConfig(filename="masters.log", level=logging.INFO)

styles_metadata, images_metadata, image_address_dictionary = load_csv()