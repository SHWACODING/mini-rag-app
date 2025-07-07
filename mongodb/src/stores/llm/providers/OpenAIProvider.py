from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
import logging


class OpenAIProvider(LLMInterface):
    """
    OpenAIProvider is a class that provides an interface to interact with OpenAI's API for text generation and embedding.
    It implements the LLMInterface and provides methods to set models, generate text, embed text, and construct prompts.
    """

    def __init__(
        self, 
        api_key: str, 
        api_url: str=None, 
        default_input_max_characters: int=1000,
        default_generation_max_output_tokens: int=1000,
        default_generation_temperature: float=0.1,
    ):
        """
        Initialize the OpenAIProvider with the given API key and optional parameters.
        
        :param api_key: The API key for OpenAI.
        :param api_url: The base URL for the OpenAI API (optional).
        :param default_input_max_characters: The default maximum number of input tokens (optional).
        :param default_generation_max_output_tokens: The default maximum number of output tokens for generation (optional).
        :param default_generation_temperature: The default temperature for generation (optional).
        """
        self.api_key = api_key
        self.api_url = api_url
        
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature
        
        self.generation_model_id = None
        
        self.embedding_model_id = None
        self.embedding_size = None
        
        self.client = OpenAI(api_key=self.api_key)
        
        if self.api_url and len(self.api_url):
            self.client.base_url = self.api_url
        
        self.enums = OpenAIEnums
        
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
            self.logger.error("Client is not initialized, Or Generation Model for OpenAI was not set.")
            return None
        
        if not self.generation_model_id:
            self.logger.error("Generation Model for OpenAI was not set.")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature
        
        chat_history.append(
            self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value)
        )
        
        response = self.client.chat.completions.create (
            model=self.generation_model_id,
            messages=chat_history,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
        
        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("Failed To Get Response From OpenAI API.")
            return None
        
        return response.choices[0].message.content
    
    def embed_text (self, text: str, document_type: str=None):
        if not self.client:
            self.logger.error("Client is not initialized, Or Embedding Model for OpenAI was not set.")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("Embedding Model for OpenAI was not set.")
            return None
        
        response = self.client.embeddings.create(
            model=self.embedding_model_id,
            input=text,
        )
        
        if not response or not response.data or len(response.data) == 0:
            self.logger.error("Failed To Get Embedding From OpenAI API.")
            return None
        
        return response.data[0].embedding
    
    def construct_prompt (self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt,
        }


