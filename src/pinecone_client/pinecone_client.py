import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, dotenv_values 

load_dotenv()

PINECONE_KEY=os.getenv("PINECONE_KEY")
INDEX_NAME=os.getenv("INDEX_NAME")

pc = Pinecone(api_key=PINECONE_KEY)