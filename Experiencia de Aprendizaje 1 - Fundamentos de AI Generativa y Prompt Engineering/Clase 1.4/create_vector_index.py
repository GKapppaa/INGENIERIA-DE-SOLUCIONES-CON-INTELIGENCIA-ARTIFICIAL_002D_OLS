from src.utils.mongodb import MongoDBClient

mongo = MongoDBClient("agent-rag-duoc-uc")
collection = mongo.get_collection("embeddings")

collection.create_search_index({
    "name": "vector_index",
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1536,
                "similarity": "cosine"
            }
        ]
    }
})

print("Vector search index created.")

