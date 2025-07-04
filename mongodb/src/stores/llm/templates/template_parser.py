import os


class TemplateParser:
    """
    A class to parse templates and extract variables.
    """

    def __init__(self, language: str = None, default_language:str = "en"):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.default_language = default_language
        self.language = None
        self.set_language(language)
        
    
    def set_language(self, language: str):
        if not language:
            language = self.default_language
        
        language_path = os.path.join(self.current_path, "locales", language)
        
        if os.path.exists(language_path):
            self.language = language
        else:
            print(f"Language '{language}' not found. Using default language '{self.default_language}'.")
            self.language = self.default_language
    
    def get_template(self, group: str, key: str, vars: dict = {}):
        if not group or not key:
            raise ValueError("Group and key must be provided.")
        
        group_path = os.path.join(self.current_path, "locales", self.language, f"{group}.py")
        
        targeted_language = self.language
        
        if not os.path.exists(group_path):
            group_path = os.path.join(self.current_path, "locales", self.default_language, f"{group}.py")
            targeted_language = self.default_language
        
        if not os.path.exists(group_path):
            raise FileNotFoundError(f"Template group '{group}' not found in language '{self.language}' or default language '{self.default_language}'.")
        
        # Import the group module dynamically
        group_module = __import__(f"stores.llm.templates.locales.{targeted_language}.{group}", fromlist=[group])
        
        if not group_module:
            raise ImportError(f"Could not import template group '{group}' in language '{targeted_language}'.")
        
        key_attribute = getattr(group_module, key)
        
        if not key_attribute:
            raise AttributeError(f"Key '{key}' not found in template group '{group}' in language '{targeted_language}'.")
        
        return key_attribute.substitute(vars)
    


