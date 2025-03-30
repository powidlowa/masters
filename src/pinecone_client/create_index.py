import logging
from datetime import datetime
from pinecone import ServerlessSpec 

# Pinecone index creation
def create_index(pc, INDEX_NAME, INDEX_DIMENSION):
    # check this index does not exist in the project environment
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME, 
            dimension=INDEX_DIMENSION,
            spec=ServerlessSpec(
                cloud='aws', # cloud provider (predefined in this case)
                region='us-east-1' # area of data storage (predefined in this case)
            )
        )
        logging.info(f'{datetime.now()}: index created')
    index = pc.Index(INDEX_NAME)
    # return created index 
    return index
