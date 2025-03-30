import os
import sys
import logging
import json
import numpy as np
from datetime import datetime

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # moves up one level to 'src'
sys.path.append(src_path) # allows to read files from sub-directories

from src.utilis.load import (
    load_csv,
    load_images
)
from src.ConvNeXt_utilis.init_convnext import (
    download_weights,
    init_convnext_model
)
from src.feature_extrtaction.feature_extraction import return_feature_extractor
from src.preprocess.preprocess_image import preprocess_image
from src.preprocess.return_embeddings import return_embeddings

logging.basicConfig(filename="masters.log", level=logging.INFO)

styles_metadata, images_metadata, image_address_dictionary = load_csv()
# get image dictionary (image_id:image_of_PIL_format)
image_set = load_images(image_address_dictionary)

# download weigts
convnext_weights = download_weights(True)
# init model
model = init_convnext_model(convnext_weights)
# init feature extractor
feature_extractor = return_feature_extractor(model)
# init image preprocesser
preprocesser = preprocess_image()

image_ids = list(image_set.keys())
# divide into batches
batches = np.array_split(image_ids, 300)

# dict to store the database embeddings
embedding_vectors = {}
counter =  0

# loop through batches
for batch in batches:
    # generate embeddings
    embeddings = return_embeddings(
        preprocesser, 
        feature_extractor,
        image_address_dictionary,
        batch,
    )
    # add new batch embeddings
    embedding_vectors.update(embeddings)
    # increment
    counter = counter + 1
    # store results to logs
    logging.info(f'{datetime.now()}: no. {counter} batch is processed')

# save embeddings of the database images to json
embeddings_json = json.dumps(embedding_vectors)
with open('embeddings/database/embeddings.json', 'w') as file:
    json.dump(embeddings_json, file)