import os
import logging
import json
import torch
import time
import numpy as np
from datetime import datetime

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
#image_set = load_images(image_address_dictionary)


#dowload weigts
#convnext_weights = download_weights(True)
#model = init_convnext_model(convnext_weights)

#feature_extractor = return_feature_extractor(model)
#preprocesser = preprocess_image()

#image_ids = list(image_set.keys())
#batches = np.array_split(image_ids, 300)

# initialize a Spark session to allow parallel run
#spark = SparkSession.builder.appName("ParallelImageProcessing").getOrCreate()

#embedding_vectors = {}
#counter =  0
#for batch in batches:
#    embeddings = return_embeddings(
#        preprocesser, 
#        feature_extractor,
#        image_address_dictionary,
#        batch,
#    )
#    embedding_vectors.update(embeddings)
#    counter = counter + 1
#    logging.info(f'{datetime.now()}: no. {counter} batch is processed')

#embeddings_json = json.dumps(embedding_vectors)
#with open('embeddings/database/embeddings.json', 'w') as file:
#    json.dump(embeddings_json, file)

with open('embeddings/database/embeddings.json', 'r') as file:
    embeddings_json = json.load(file)
embeddings_dict = json.loads(embeddings_json)

batches = np.array_split(list(embeddings_dict.keys()), 300)
pc = init_pinecone()
index = create_index(pc, INDEX_NAME, INDEX_DIMENSION)


MAX_RETRIES = 3 
counter = 0
for batch in batches:
    retries = 0
    while retries < MAX_RETRIES:
        try:
            chunked_dictionary = {str(picture_id): embeddings_dict[picture_id] for picture_id in batch}
            upsert_index(index, chunked_dictionary, styles_metadata)
            counter += 1
            logging.info(f'{datetime.now()}: Batch {counter} processed successfully')
            break 
        except Exception as e:
            logging.warning(f'{datetime.now()}: Batch {counter + 1} failed on attempt {retries + 1}')
            logging.warning(f'Error: {e}')
            retries += 1
            time.sleep(2) 
    else:
        logging.info(f'{datetime.now()}: Batch {counter + 1} could not be processed after {MAX_RETRIES} retries')
        logging.info(f'{datetime.now()}: Starting image-by-image processing')
        for picture_id in batch:
            try:
                upsert_index(index, {str(picture_id): embeddings_dict[picture_id]}, styles_metadata)
            except Exception as e:
                logging.warning(f'{datetime.now()}: Image ID {picture_id} cannot be processed. Error: {e}')
        counter += 1