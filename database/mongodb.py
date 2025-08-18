"""
MongoDB database connection and session management.
"""
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from .config import MONGODB_DATABASE, MONGODB_URL


client = MongoClient(MONGODB_URL, server_api=ServerApi("1"))

try:
    client.admin.command("ping")
    print("Pinged your deployment. Successfully connected to MongoDB!")
    database = client.get_database(MONGODB_DATABASE)

except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
