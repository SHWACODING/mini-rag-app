from abc import ABC, abstractmethod
from typing import List
from models.db_schemes import RetrievedDocument


class VectorDBInterface(ABC):
    
    @abstractmethod
    def connect (self):
        """Connect to the vector database."""
        pass
    
    @abstractmethod
    def disconnect (self):
        """Disconnect from the vector database."""
        pass
    
    @abstractmethod
    def is_collection_existed (self, collection_name: str) -> bool:
        """Check if a collection exists in the vector database."""
        pass
    
    @abstractmethod
    def list_all_collections (self) -> List:
        """List all collections in the vector database."""
        pass
    
    @abstractmethod
    def get_collection_info (self, collection_name: str) -> dict:
        """Get information about a collection in the vector database."""
        pass
    
    @abstractmethod
    def delete_collection (self, collection_name: str):
        """Delete a collection from the vector database."""
        pass
    
    @abstractmethod
    def create_collection (self, collection_name: str, embedding_size: int, do_reset: bool = False):
        """Create a new collection in the vector database."""
        pass
    
    @abstractmethod
    def insert_one (
        self, 
        collection_name: str, 
        text: str, 
        vector: list, 
        metadata: dict = None, 
        record_id: str = None
    ):
        """Insert a single record into a collection."""
        pass
    
    @abstractmethod
    def insert_many (
        self, 
        collection_name: str, 
        texts: List, 
        vectors: List, 
        metadata: List = None, 
        record_ids: List = None, 
        batch_size: int = 50
    ):
        """Insert multiple records into a collection."""
        pass
    
    @abstractmethod
    def search_by_vector (self, collection_name: str, vector: list, limit: int = 5) -> List[RetrievedDocument]:
        """Search for records in a collection by vector."""
        pass
