import logging
import torch
from PIL import Image
from datetime import datetime

def return_embeddings(
        preprocesser, 
        feature_extractor,
        image_address_dictionary,
        image_ids,
):
    embedding_vectors = {}
    for image_id in image_ids:
        # load and preprocess an image
        image_path = image_address_dictionary[image_id]
        image = Image.open(image_path)
        # add batch dimension
        input_tensor = preprocesser(image).unsqueeze(0) 
        with torch.no_grad():
            embedding = feature_extractor(input_tensor)
        # flatten the embedding to get a 1D vector
        embedding_vectors[int(image_id)] = embedding.flatten()
    logging.info(f'{datetime.now()}: embeddings generated')

    return embedding_vectors
