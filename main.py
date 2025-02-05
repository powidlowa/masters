import os
import random
import uuid
import logging
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

INDEX_NAME=os.getenv("INDEX_NAME")

#dowload weigts
convnext_weights = download_weights(True)
model = init_convnext_model(convnext_weights)

feature_extractor = return_feature_extractor(model)
preprocesser = preprocess_image()

images = os.listdir("data/query/images")
query_image = f"data/query/images/{random.choice(images)}"
generated_id = uuid.uuid4().int

image_address_dictionary = {generated_id:query_image}
embedding_query = return_embeddings(
    preprocesser, 
    feature_extractor,
    image_address_dictionary,
    [generated_id]
)
print(embedding_query)
pc = init_pinecone()
index = pc.Index(INDEX_NAME)
response = index.query(vector=embedding_query[generated_id], top_k=4) #include_metadata=True)
print(response)