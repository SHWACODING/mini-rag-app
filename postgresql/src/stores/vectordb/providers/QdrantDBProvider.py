from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from qdrant_client import models, QdrantClient
import logging
from typing import List
from models.db_schemes import RetrievedDocument


class QdrantDBProvider(VectorDBInterface):
    def __init__(self, db_client: str, default_vector_size: int = 768, distance_method: str = None, index_threshold: int = 100):
        
        self.client = None
        self.db_client = db_client
        self.distance_method = None
        self.default_vector_size = default_vector_size
        
        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT_PRODUCT.value:
            self.distance_method = models.Distance.DOT
        else:
            self.distance_method = models.Distance.EUCLID
        
        self.logger = logging.getLogger("uvicorn")
        
    async def connect(self):
        self.client = QdrantClient(path=self.db_client)
        self.logger.info(f"Connected to QdrantDB at {self.db_client}")
    
    async def disconnect(self):
        self.client = None
        self.logger.info(f"Disconnected from QdrantDB at {self.db_client}")
    
    async def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)
    
    async def list_all_collections(self) -> List:
        return self.client.get_collections()
    
    async def get_collection_info (self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name=collection_name)

    async def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            self.logger.info(f"Deleting/Resetting QDRANT collection: {collection_name}")
            return self.client.delete_collection(collection_name=collection_name)
        else:
            self.logger.info(f"Collection not found: {collection_name}", )
            return None
    
    async def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False) -> bool:
        if do_reset:
            self.delete_collection(collection_name=collection_name)
        
        if not self.is_collection_existed(collection_name):
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size, 
                        distance=self.distance_method
                    )
                )
                self.logger.info(f"Qdrant Collection: {collection_name} Created Successfully!!!!")
                return True
            except Exception as e:
                self.logger.error(f"Error creating collection: {collection_name}")
                return False
        else:
            self.logger.info(f"Collection: {collection_name} Already Exists!")
            return True
    
    async def insert_one (
        self, 
        collection_name: str, 
        text: str, 
        vector: list, 
        metadata: dict = None, 
        record_id: str = None
    ):
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can't Insert New Record To Non-existed Collection: {collection_name}", )
            return False
        
        try:
            _ = self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=[record_id],
                        vector=vector,
                        payload={
                            "text": text,
                            "metadata": metadata
                        }
                    )
                ]
            )
        except Exception as e:
            self.logger.error(f"Insert New Record Failed: {e}")
            return False
        
        return True
    
    async def insert_many (
        self, 
        collection_name: str, 
        texts: List, 
        vectors: List, 
        metadata: List = None, 
        record_ids: List = None, 
        batch_size: int = 50
    ):
        if metadata is None:
            metadata = [None] * len(texts)
        
        if record_ids is None:
            record_ids = list(range(0, len(texts)))
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can't Insert New Records To Non-existed Collection: {collection_name}")
            return False
        
        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size
            
            batch_texts = texts[i : batch_end]
            batch_vectors = vectors[i : batch_end]
            batch_metadata = metadata[i : batch_end]
            batch_record_ids = record_ids[i : batch_end]
            
            batch_records = [
                models.Record(
                    id=batch_record_ids[x],
                    vector=batch_vectors[x],
                    payload={
                        "text": batch_texts[x],
                        "metadata": batch_metadata[x]
                    }
                )
                
                for x in range(len(batch_texts))
            ]
            
            try:
                _ = self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records
                )
            except Exception as e:
                self.logger.error(f"Error while inserting records: {e}")
                return False
        
        return True
    
    async def search_by_vector (self, collection_name: str, vector: list, limit: int = 5):
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
        )
        
        if not results or len(results) == 0:
            self.logger.error("No results found for the given vector.")
            return None
        
        return [
            RetrievedDocument(**{
                "score": result.score,
                "text": result.payload["text"],
            }) for result in results
        ]
        
    
