from ..LLMInterface import LLMInterface
from ..LLMEnums import GeminiEnums
import logging
import google.generativeai as genai


class GeminiProvider(LLMInterface):
    """
    GeminiProvider is a class that provides an interface to interact with Google's Gemini API
    for text generation and embedding. It implements the LLMInterface and provides methods
    to set models, generate text, embed text, and construct prompts.
    """

    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):
        """
        Initialize the GeminiProvider with the given API key and optional parameters.

        :param api_key: The API key for Google Gemini.
        :param default_input_max_characters: Max number of characters in input prompt.
        :param default_generation_max_output_tokens: Max number of output tokens.
        :param default_generation_temperature: Generation randomness/creativity.
        """
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.enums = GeminiEnums
        self.logger = logging.getLogger(__name__)
        genai.configure(api_key=self.api_key)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list = None, max_output_tokens: int = None, temperature: float = None):
        if not self.generation_model_id:
            self.logger.error("Generation Model for Gemini was not set.")
            return None

        try:
            model = genai.GenerativeModel(model_name=self.generation_model_id)
            chat = model.start_chat(history=chat_history)

            max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
            temperature = temperature or self.default_generation_temperature
            
            response = chat.send_message(
            self.process_text(prompt),
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            })

            return response.text
        except Exception as e:
            self.logger.error(f"Failed to get response from Gemini API: {e}")
            return None

    def embed_text(self, text: str, document_type: str = None):
        if not self.embedding_model_id:
            self.logger.error("Embedding Model for Gemini was not set.")
            return None

        try:
            response = genai.embed_content(
                model=self.embedding_model_id,
                content=text,
                task_type="retrieval_document" if document_type == "document" else "retrieval_query",
            )

            return response["embedding"]
        except Exception as e:
            self.logger.error(f"Failed to get embedding from Gemini API: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "parts": [prompt],
        }