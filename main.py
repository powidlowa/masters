import os
import json
import time
import uuid
import logging
import argparse
from datetime import datetime

from src.ConvNeXt_utilis.init_convnext import (
    download_weights,
    init_convnext_model
)
from src.feature_extrtaction.feature_extraction import return_feature_extractor
from src.preprocess.preprocess_image import preprocess_image
from src.preprocess.return_embeddings import return_embeddings
from src.pinecone_client.init_pinecone import init_pinecone

# configure loggin
logging.basicConfig(filename="masters.log", level=logging.INFO)

# start point to calculate runtime
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--gender', help='gender of the search: male, female, boys, girls, unisex')
parser.add_argument('--quantity', help='quantity of imgaes to process')
args = parser.parse_args()

# get index name
INDEX_NAME=os.getenv("INDEX_NAME")

# dowload weigts
convnext_weights = download_weights(True)
# init model
model = init_convnext_model(convnext_weights)
# init feature extractor
feature_extractor = return_feature_extractor(model)
# init preprocessor
preprocesser = preprocess_image()

# map arguments to the gender to parse query images
mapping_gender = {
    'female':'Women',
    'male':'Men',
    'unisex':'Unisex',
    'boys':'Boys',
    'girls':'Girls',
    'test_of_speed': 'Women'
}
responses = {}
images = os.listdir(f"data/query/images/{args.gender}")[:int(args.quantity)]

for image in images:
    if image != '.DS_Store':
        query_image = f"data/query/images/{args.gender}/{image}" # read query image
        generated_id = uuid.uuid4().int # generate id for the response json
        # assign image to generatedn id
        image_address_dictionary = {generated_id:query_image}
        # return vector representation of the query image
        embedding_query = return_embeddings(
            preprocesser, 
            feature_extractor,
            image_address_dictionary,
            [generated_id]
        )
        # initialize pinecone API
        pc = init_pinecone()
        # checkout needed index name
        index = pc.Index(INDEX_NAME)
        # using Pinoce API find best matches
        response = index.query(
            vector=embedding_query[generated_id], # pass vector of the query image
            top_k=4, # four bes matches to be returned
            include_values=False, # do not include vectors to the response
            includeMetadata=True, # include metadata to the response
            filter={'gender': mapping_gender[args.gender]} # meta-data pre-filtering (here: gender)
        ).to_dict() # store to dictionary
        response["query_image"] = query_image
        print(query_image)
        response["generated_id"] = generated_id
        # save results to json file
        with open(f'data/query/responses/{args.gender}/{generated_id}_response.json', 'w') as file:
            json.dump(response, file)

# end of runtimne tracking
end_time = time.time()
print(f"Execution time for {len(images)} images: {end_time - start_time:.6f} seconds")
