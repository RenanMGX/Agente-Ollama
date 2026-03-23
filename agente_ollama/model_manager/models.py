import os
import ollama

from typing import Literal, Dict, List
from dotenv import load_dotenv; load_dotenv()

class Models:
    list_models: Dict[str, List[str]] = {}
    
    @property
    def completion(self):
        if self.__completion:
            return self.__completion
        raise ValueError("COMPLETION_MODEL is not set.")
    @completion.setter
    def completion(self, value:str) -> None:
        if self.check_model_capability(value, capability='completion'):
            self.__completion = value
            self.__completetion_capability(value)
        else:
            raise ValueError(f"Model '{value}' does not exist.")
        
    @property
    def tools(self):
        if self.__tools:
            return self.__tools
        raise ValueError("TOOLS_MODEL is not set.")
    @tools.setter
    def tools(self, value:str) -> None:
        if self.check_model_capability(value, capability='tools'):
            self.__tools = value
            self.__completetion_capability(value)
        else:
            raise ValueError(f"Model '{value}' does not exist.")
        
    @property
    def thinking(self):
        if self.__thinking:
            return self.__thinking
        raise ValueError("THINKING_MODEL is not set.")
    @thinking.setter
    def thinking(self, value:str) -> None:
        if self.check_model_capability(value, capability='thinking'):
            self.__thinking = value
            self.__completetion_capability(value)
        else:
            raise ValueError(f"Model '{value}' does not exist.")
        
    @property
    def vision(self):
        if self.__vision:
            return self.__vision
        raise ValueError("VISION_MODEL is not set.")
    @vision.setter
    def vision(self, value:str) -> None:
        if self.check_model_capability(value, capability='vision'):
            self.__vision = value
            self.__completetion_capability(value)
        else:
            raise ValueError(f"Model '{value}' does not exist.")
        
    @property
    def embedding(self):
        if self.__embedding:
            return self.__embedding
        raise ValueError("EMBEDDING_MODEL is not set.")
    @embedding.setter
    def embedding(self, value:str) -> None:
        if self.check_model_capability(value, capability='embedding'):
            self.__embedding = value
            self.__completetion_capability(value)
        else:
            raise ValueError(f"Model '{value}' does not exist.")
        
    
    @staticmethod
    def model_exists(model_name:str) -> bool:
        if model_name in Models.list_models:
            return True
        return False
    
    @staticmethod
    def check_model_capability(
        model_name:str, 
        *, 
        capability:Literal["completion", "tools", "thinking", "vision", "embedding"]
    ) -> bool:
        if Models.model_exists(model_name):
            if capability in Models.list_models[model_name]:
                return True
        else:
            raise ValueError(f"Model '{model_name}' does not exist.")
        return False
            
    def __completetion_capability(self, model_name:str):
        if self.check_model_capability(model_name, capability='completion'):
            if not self.__completion:
                self.__completion = model_name
    
        if self.check_model_capability(model_name, capability='embedding'):
            if not self.__embedding:
                self.__embedding = model_name
    
        if self.check_model_capability(model_name, capability='tools'):
            if not self.__tools:
                self.__tools = model_name
    
        if self.check_model_capability(model_name, capability='thinking'):
            if not self.__thinking:
                self.__thinking = model_name
    
        if self.check_model_capability(model_name, capability='vision'):
            if not self.__vision:
                self.__vision = model_name
    
    def __str__(self):
        return f"Models(completion='{self.__completion}', tools='{self.__tools}', thinking='{self.__thinking}', vision='{self.__vision}', embedding='{self.__embedding}')"
    
    def __repr__(self):
        return f"Models(completion='{self.__completion}', tools='{self.__tools}', thinking='{self.__thinking}', vision='{self.__vision}', embedding='{self.__embedding}')"
    
    def __init__(self):
        self.get_all_models(force_update=True)
        
        self.__completion = ""
        self.__tools = ""
        self.__thinking = ""
        self.__vision = ""
        self.__embedding = ""
        
        if (model:=os.getenv("COMPLETION_MODEL")):
            self.completion = model
        if (model:=os.getenv("TOOLS_MODEL")):
            self.tools = model
        if (model:=os.getenv("THINKING_MODEL")):
            self.thinking = model
        if (model:=os.getenv("VISION_MODEL")):
            self.vision = model
        if (model:=os.getenv("EMBEDDING_MODEL")):
            self.embedding = model
                
    def get_all_models(self, *, force_update:bool=False) -> Dict[str, List[str]]:
        if force_update or not Models.list_models:
            for model in ollama.list().models:
                model_name = str(model.model)
                if model_name:
                    Models.list_models[model_name] = [x for x in valis_list] if (valis_list:=ollama.show(model_name).capabilities) else []
        return Models.list_models
        
            
        
if __name__ == "__main__":
    bot = Models()
    #bot.completion = "qwen2.5:3b"
    #bot.thinking = ""
    bot.embedding = "granite-embedding:278m"
    print(
        bot,
        bot.completion,
        bot.embedding
    )
    