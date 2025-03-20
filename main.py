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

logging.basicConfig(filename="masters.log", level=logging.INFO)

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--gender', help='gender of the search: male, female, boys, girls, unisex')
parser.add_argument('--quantity', help='quantity of imgaes to process')
args = parser.parse_args()

INDEX_NAME=os.getenv("INDEX_NAME")

#dowload weigts
convnext_weights = download_weights(True)
model = init_convnext_model(convnext_weights)

feature_extractor = return_feature_extractor(model)
preprocesser = preprocess_image()

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
        query_image = f"data/query/images/{args.gender}/{image}"
        generated_id = uuid.uuid4().int

        image_address_dictionary = {generated_id:query_image}
        embedding_query = return_embeddings(
            preprocesser, 
            feature_extractor,
            image_address_dictionary,
            [generated_id]
        )
        pc = init_pinecone()
        index = pc.Index(INDEX_NAME)
        response = index.query(
            vector=embedding_query[generated_id], 
            top_k=4,
            include_values=False,
            includeMetadata=True,
            filter={'gender': mapping_gender[args.gender]}
        ).to_dict()
        response["query_image"] = query_image
        print(query_image)
        response["generated_id"] = generated_id
        with open(f'data/query/responses/{args.gender}/{generated_id}_response.json', 'w') as file:
            json.dump(response, file)

end_time = time.time()
print(f"Execution time for {len(images)} images: {end_time - start_time:.6f} seconds")
