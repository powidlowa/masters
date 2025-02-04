import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, dotenv_values 

load_dotenv()

PINECONE_KEY=os.getenv("PINECONE_KEY")

def init_pinecone(PINECONE_KEY=PINECONE_KEY):
    return Pinecone(api_key=PINECONE_KEY)
