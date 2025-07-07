from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
import cohere
import logging


class CoHereProvider(LLMInterface):
    """
    CoHereProvider is a class that provides an interface to interact with CoHere's API for text generation and embedding.
    It implements the LLMInterface and provides methods to set models, generate text, embed text, and construct prompts.
    """

    def __init__(
        self, 
        api_key: str, 
        default_input_max_characters: int=1000,
        default_generation_max_output_tokens: int=1000,
        default_generation_temperature: float=0.1,
    ):
        """
        Initialize the CoHereProvider with the given API key and optional parameters.

        :param api_key: The API key for CoHere.
        :param default_input_max_characters: The default maximum number of input tokens (optional).
        :param default_generation_max_output_tokens: The default maximum number of output tokens for generation (optional).
        :param default_generation_temperature: The default temperature for generation (optional).
        """
        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None
        
        self.client = cohere.Client(api_key=self.api_key)
        
        self.enums = CoHereEnums
        
        self.logger = logging.getLogger(__name__)

    
    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
    
    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
    
    # Custom Function To Process Text
    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()
    
    def generate_text (self, prompt: str, chat_history: list=[], max_output_tokens: int=None, temperature: float=None):
        if not self.client:
            self.logger.error("Cohere client is not initialized.")
            return None
        
        if not self.generation_model_id:
            self.logger.error("Generation model ID is not set.")
            return None
        
        # Ensure chat_history is in the correct format
        formatted_chat_history = []
        for item in chat_history:
            if isinstance(item, dict) and "message" in item:
                formatted_chat_history.append(item)
            elif isinstance(item, str):
                formatted_chat_history.append({"role": "USER", "message": item})
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature
        
        response = self.client.chat (
            model=self.generation_model_id,
            chat_history=formatted_chat_history,
            message=self.process_text(prompt),
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        
        if not response or not response.text:
            self.logger.error("No response from CoHere API.")
            return None
        
        return response.text
    
    def embed_text (self, text: str, document_type: str=None):
        if not self.client:
            self.logger.error("Cohere client is not initialized.")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model ID is not set.")
            return None
        
        input_type = CoHereEnums.DOCUMENT
        if document_type == DocumentTypeEnum.QUERY:
            input_type = CoHereEnums.QUERY
        
        response = self.client.embed (
            model=self.embedding_model_id,
            texts=[self.process_text(text)],
            input_type=input_type,
            embedding_types=['float']
        )
        
        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("Failed to get embedding from CoHere API.")
            return None
        
        return response.embeddings.float[0]
    
    def construct_prompt (self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt,
        }
