"""
Chromadb database connection and session management.
"""
import chromadb
from chromadb.utils import embedding_functions

from .config import (CHROMA_API_KEY, CHROMA_COLLECTION, CHROMA_DATABASE,
                     CHROMA_TENANT)
from .utils.categories import get_categories

chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="AryehRotberg/ToS-Sentence-Transformers-V3"
)

collection = chroma_client.get_or_create_collection(
    CHROMA_COLLECTION, embedding_function=sentence_transformer_ef
)

def add_categories() -> None:
    categories = get_categories()

    collection.add(
        documents=categories,
        metadatas=[{"category": cat} for cat in categories],
        ids=[str(i) for i in range(len(categories))]
    )
