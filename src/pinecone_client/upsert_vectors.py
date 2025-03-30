import logging
from datetime import datetime

# upserting (aka ingestion) of the vectors to the database
def upsert_index(index, embedding_vectors, styles_metadata):
    vectors_to_upsert = []
    metadata_dictionary = styles_metadata.T.to_dict()
    # loop through passed vectors (executed in batches)
    for k,v in embedding_vectors.items():
        tmp_dict = {
            "id": str(k), # image_id
            "values": v, # vector of the image
            "metadata": metadata_dictionary[int(k)] # meta-data of the image
        }
        vectors_to_upsert.append(tmp_dict)

    index.upsert(vectors=vectors_to_upsert)
    logging.info(f'{datetime.now()}: {len(vectors_to_upsert)} vectors upserted')