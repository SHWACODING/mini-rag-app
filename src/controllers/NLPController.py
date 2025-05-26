from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json

class NLPController(BaseController):
    def __init__(self, vector_db_client, generation_client, embedding_client):
        super().__init__()
        
        self.vector_db_client = vector_db_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
    
    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vector_db_client.delete_collection(collection_name=collection_name)
    
    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vector_db_client.get_collection_info(collection_name=collection_name)
        
        return json.loads(
            json.dumps(collection_info, default=lambda o: o.__dict__)
        )
    
    def index_into_vector_db (self, project: Project, chunks: List[DataChunk], chunks_ids: List[int], do_reset: bool = False):
        
        # Step 1: Get Collection Name
        collection_name = self.create_collection_name(project_id=project.project_id)
        
        # Step 2: Manage Items
        texts = [ c.chunk_text for c in chunks ]
        metadata = [ c.chunk_metadata for c in chunks ]
        
        vectors = [
            self.embedding_client.embed_text(text=text, document_type=DocumentTypeEnum.DOCUMENT.value) for text in texts
        ]
        
        # Step 3: Create Collection if Not Exists
        _ = self.vector_db_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset
        )
        
        # Step 4: Insert Into Vector DB
        _ = self.vector_db_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadata=metadata,
            record_ids=chunks_ids
        )
        
        return True
    
    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):
        
        # Step 1: Get Collection Name
        collection_name = self.create_collection_name(project_id=project.project_id)
        
        # Step 2: Embed Text or Get Text Embedding Vector
        vector = self.embedding_client.embed_text(
            text=text, 
            document_type=DocumentTypeEnum.QUERY.value
        )
        
        if not vector or len(vector) == 0:
            self.logger.error("Failed to embed the search text.")
            return False
        
        # Step 3: Do Semantic Search
        results = self.vector_db_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit
        )
        
        if not results or len(results) == 0:
            self.logger.error("No results found in the vector database.")
            return False
        
        return json.loads(
            json.dumps(results, default=lambda o: o.__dict__)
        )


