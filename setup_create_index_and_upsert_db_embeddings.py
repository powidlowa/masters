import os
import logging
import json
import time
import numpy as np
from datetime import datetime

from src.pinecone_client.init_pinecone import init_pinecone
from src.pinecone_client.create_index import create_index
from src.pinecone_client.upsert_vectors import upsert_index
from src.utilis.load import load_csv

logging.basicConfig(filename="masters.log", level=logging.INFO)

INDEX_DIMENSION=1024 # defined based on convnext "base" model setting for embeggings
INDEX_NAME=os.getenv("INDEX_NAME")

# load metadata from csv files
styles_metadata, images_metadata, image_address_dictionary = load_csv()

# open json file with embeddings of the database images
with open('embeddings/database/embeddings.json', 'r') as file:
    embeddings_json = json.load(file)
embeddings_dict = json.loads(embeddings_json)

# split task into 300 batches
batches = np.array_split(list(embeddings_dict.keys()), 300)
# init Pionoce API
pc = init_pinecone()
# create a vetctor index in Pinecone
index = create_index(pc, INDEX_NAME, INDEX_DIMENSION)


MAX_RETRIES = 3 
counter = 0
# loop through batches
for batch in batches:
    retries = 0
    # not more than 3 retries allowed, if no successes - log which batch failed
    while retries < MAX_RETRIES: 
        try:
            # create dict with the respect to the batch (image_id_image_vector)
            chunked_dictionary = {str(picture_id): embeddings_dict[picture_id] for picture_id in batch}
            # upsert vecor tio the database along with metadata
            upsert_index(index, chunked_dictionary, styles_metadata)
            counter += 1
            logging.info(f'{datetime.now()}: Batch {counter} processed successfully')
            break 
        except Exception as e: # log expection and current attempt count
            logging.warning(f'{datetime.now()}: Batch {counter + 1} failed on attempt {retries + 1}')
            logging.warning(f'Error: {e}')
            retries += 1
            time.sleep(2) 
    else:
        logging.info(f'{datetime.now()}: Batch {counter + 1} could not be processed after {MAX_RETRIES} retries')
        logging.info(f'{datetime.now()}: Starting image-by-image processing')
        # if batch fails, loop through every image to upload one by one
        # otherwise the whole batch will not be ingested
        for picture_id in batch:
            try:
                upsert_index(index, {str(picture_id): embeddings_dict[picture_id]}, styles_metadata)
            except Exception as e:
                logging.warning(f'{datetime.now()}: Image ID {picture_id} cannot be processed. Error: {e}')
        counter += 1