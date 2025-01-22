import logging
from datetime import datetime

def upsert_index(index, embedding_vectors, styles_metadata):
    vectors_to_upsert = []
    metadata_dictionary = styles_metadata.T.to_dict()
    for k,v in embedding_vectors.items():
        tmp_dict = {
            "id": str(k),
            "values": v,
            "metadata": metadata_dictionary[k]
        }
        vectors_to_upsert.append(tmp_dict)

    index.upsert(vectors=vectors_to_upsert)
    logging.info(f'{datetime.now()}: len(vectors_to_upsert) vectors upserted')