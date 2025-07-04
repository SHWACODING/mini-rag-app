from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json
import logging

logger = logging.getLogger('uvicorn.error')

class NLPController(BaseController):
    def __init__(self, vector_db_client, generation_client, embedding_client, template_parser):
        super().__init__()
        
        self.vector_db_client = vector_db_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
    
    def create_collection_name(self, project_id: str):
        return f"collection_{self.vector_db_client.default_vector_size}_{project_id}".strip()
    
    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vector_db_client.delete_collection(collection_name=collection_name)
    
    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vector_db_client.get_collection_info(collection_name=collection_name)
        
        return json.loads(
            json.dumps(collection_info, default=lambda o: o.__dict__)
        )
    
    async def index_into_vector_db (self, project: Project, chunks: List[DataChunk], chunks_ids: List[int], do_reset: bool = False):
        
        # Step 1: Get Collection Name
        collection_name = self.create_collection_name(project_id=project.project_id)
        
        # Step 2: Manage Items
        texts = [ c.chunk_text for c in chunks ]
        metadata = [ c.chunk_metadata for c in  chunks]
        
        vectors = self.embedding_client.embed_text(text=texts, document_type=DocumentTypeEnum.DOCUMENT.value)
        
        # Step 3: Create Collection if Not Exists
        _ = await self.vector_db_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset
        )
        
        # Step 4: Insert Into Vector DB
        _ = await self.vector_db_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids
        )
        
        return True
    
    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):
        
        # Step 1: Get Collection Name
        
        query_vector = None
        
        collection_name = self.create_collection_name(project_id=project.project_id)
        
        # Step 2: Embed Text or Get Text Embedding Vector
        vectors = self.embedding_client.embed_text(
            text=text, 
            document_type=DocumentTypeEnum.QUERY.value
        )
        
        if not vectors or len(vectors) == 0:
            logger.error("Failed to embed the search text.")
            return False
        
        if isinstance(vectors, list) and len(vectors) > 0:
            query_vector = vectors[0]
        
        if not query_vector:
            logger.error("No valid vector found for the search text.")
            return False
        
        # Step 3: Do Semantic Search
        results = await self.vector_db_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=limit
        )
        
        if not results or len(results) == 0:
            logger.error("No results found in the vector database.")
            return False
        
        return results
    
    async def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        answer, full_prompt, chat_history = None, None, None

        # Step 1: Retrieve related documents
        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        if not retrieved_documents:
            return answer, full_prompt, chat_history

        # Step 2: Construct LLM prompt components
        system_prompt = self.template_parser.get_template("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get_template(
                "rag",
                "document_prompt",
                {
                    "doc_num": idx + 1,
                    "chunk_text": self.generation_client.process_text(doc.text),
                }
            )
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get_template(
            "rag",
            "footer_prompt",
            {"query": query}
        )

        # Step 3: Merge all prompt parts into a single user message (Gemini-compatible)
        full_prompt = "\n\n".join([
            system_prompt,        # Embed system-level instructions directly
            documents_prompts,
            footer_prompt
        ])

        chat_history = []  # No 'system' role allowed in Gemini — start clean

        # Step 4: Get response from LLM
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        if not answer:
            logger.error("Failed to generate an answer from the LLM.")
            return None, full_prompt, chat_history

        return answer, full_prompt, chat_history
