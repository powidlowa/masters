import logging
from src.utilis.load import (
    load_csv,
    load_images
)
from src.ConvNeXt_utilis.init_convnext import (
    download_weights,
    init_convnext_model
)
from src.feature_extrtaction.feature_extraction import (
    return_feature_extractor
)
from src.preprocess.preprocess_image import (
    preprocess_image
)
from src.preprocess.return_embeddings import (
    return_embeddings
)

logging.basicConfig(filename="masters.log", level=logging.INFO)

styles_metadata, images_metadata, image_address_dictionary = load_csv()
image_set = load_images(image_address_dictionary)

convnext_weights = download_weights()
model = init_convnext_model(convnext_weights)
feature_extractor = return_feature_extractor(model)
preprocesser = preprocess_image()
image_ids = list(image_address_dictionary.keys())[:20]
embeddings = return_embeddings(
        preprocesser, 
        feature_extractor,
        image_address_dictionary,
        image_ids,
)

