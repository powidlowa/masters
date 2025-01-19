import logging
from src.utilis.load import (
    load_csv,
    load_images
)
from src.ConvNeXt_utilis.init_convnext import (
    download_weights,
    init_convnext_model
)

logging.basicConfig(filename="masters.log", level=logging.INFO)

styles_metadata, images_metadata, image_address_dictionary = load_csv()
image_set = load_images(image_address_dictionary)

convnext_weights = download_weights()
model = init_convnext_model(convnext_weights)