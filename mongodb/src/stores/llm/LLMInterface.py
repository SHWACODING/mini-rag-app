from abc import ABC, abstractmethod


class LLMInterface (ABC):
    
    @abstractmethod
    def set_generation_model (self, model_id: str):
        """
        Set the generation model to be used.
        :param model_id: The ID of the model to be used for generation.
        """
        pass
    
    @abstractmethod
    def set_embedding_model (self, model_id: str, embedding_size: int):
        """
        Set the embedding model to be used.
        :param model_id: The ID of the model to be used for embedding.
        :param embedding_size: The size of the embedding (optional).
        """
        pass
    
    @abstractmethod
    def generate_text (self, prompt: str, chat_history: list=[], max_output_tokens: int=None, temperature: float=None):
        """
        Generate text based on the given prompt.
        
        :param prompt: The input prompt for text generation.
        :param chat_history: The history of the conversation (optional).
        :param max_output_tokens: The maximum number of tokens to generate.
        :param temperature: The temperature for sampling. Higher values result in more random outputs.
        :return: The generated text.
        """
        pass
    
    @abstractmethod
    def embed_text (self, text: str, document_type: str=None):
        """
        Embed the given text.
        
        :param text: The input text to be embedded.
        :param document_type: The type of document (e.g., "text", "image").
        :return: The embedded representation of the text.
        """
        pass
    
    @abstractmethod
    def construct_prompt (self, prompt: str, role: str):
        """
        Construct a prompt for the LLM.
        
        :param prompt: The input prompt.
        :param role: The role of the model (e.g., "user", "assistant").
        :return: The constructed prompt.
        """
        pass

