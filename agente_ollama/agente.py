from .model_manager.models import Models

class Agente:
    @property
    def models(self) -> Models:
        return self.__models
    @models.setter
    def models(self, value:Models) -> None:
        if isinstance(value, Models):
            self.__models = value
        else:
            raise ValueError("models must be an instance of Models class.")
    
    def __init__(self, model:Models=Models()):
        self.__models:Models = model  
        
        
if __name__ == "__main__":
    bot = Agente()
    print(bot.models)
