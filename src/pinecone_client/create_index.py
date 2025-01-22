import os
from pinecone import ServerlessSpec 

def create_index(pc, INDEX_NAME, INDEX_DIMENSION):
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME, 
            dimension=INDEX_DIMENSION,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    index = pc.Index(INDEX_NAME)

    return index
