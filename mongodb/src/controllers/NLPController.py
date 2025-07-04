from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json
import logging

class NLPController(BaseController):
    def __init__(self, vector_db_client, generation_client, embedding_client, template_parser):
        super().__init__()
        
        self.vector_db_client = vector_db_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.logger = logging.getLogger(__name__)
    
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
        
        return results
    
    def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        # Step 1: Search Vector DB
        retrieved_documents = self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit
        )
        
        if not retrieved_documents or len(retrieved_documents) == 0:
            self.logger.error("Failed to search the vector database.")
            return None, None, None
        
        # Step 2: Construct LLM Prompt
        system_prompt = self.template_parser.get_template("rag", "system_prompt")
        
        documents_prompts = "\n".join([
            self.template_parser.get_template(
                "rag", 
                "document_prompt", 
                vars={
                    "doc_num": idx + 1, 
                    "chunk_text": self.generation_client.process_text(doc.text)
                }
            ) for idx, doc in enumerate(retrieved_documents)
        ])
        
        footer_prompt = self.template_parser.get_template(
            "rag", 
            "footer_prompt",
            vars={
                "query": query
            }
        )
        
        # Combine all prompts into a single user message
        full_prompt = f"{system_prompt}\n\n{documents_prompts}\n\n{footer_prompt}"
        
        # Create a simple chat history with just the user message
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=full_prompt,
                role=self.generation_client.enums.USER.value,
            )
        ]
        
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history,
        )
        
        if not answer:
            self.logger.error("Failed to generate an answer using the LLM.")
            return None, full_prompt, chat_history
        
        return answer, full_prompt, chat_history
        

