from langsmith import traceable
from src.utils.embeddings import EmbeddingClient
from src.utils.mongodb import MongoDBClient


class Retriever:
    def __init__(self, db_name: str = "agent-rag-duoc-uc", collection_name: str = "embeddings", index_name: str = "vector_index"):
        self.embedder = EmbeddingClient()
        self.mongo = MongoDBClient(db_name)
        self.collection = self.mongo.get_collection(collection_name)
        self.index_name = index_name

    @traceable(name="retrieve")
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.get_embedding(query)

        results = self.collection.aggregate([
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ])

        return list(results)