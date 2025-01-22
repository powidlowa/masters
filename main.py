import os
import logging
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
from src.pinecone_client.init_pinecone import init_pinecone
from src.pinecone_client.create_index import create_index
from src.pinecone_client.upsert_vectors import upsert_index

logging.basicConfig(filename="masters.log", level=logging.INFO)

INDEX_DIMENSION=1024 # defined based on convnext "base" model setting for embeggings
INDEX_NAME=os.getenv("INDEX_NAME")

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
pc = init_pinecone()
index = create_index(pc, INDEX_NAME, INDEX_DIMENSION)
upsert_index(index, embeddings, styles_metadata)


