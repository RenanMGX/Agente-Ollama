import ollama
import numpy as np

from typing import Any, Callable, Literal, Mapping, Sequence, List, Dict, Union, Optional, Tuple
from copy import deepcopy
from datetime import date, datetime

from ollama._types import (
  ChatResponse,
  Tool,
)


class Embedding():
    @property
    def embeddings_base(self) -> Dict[str, Any]| None:
        return self.__embeddings_base
    @embeddings_base.setter
    def embeddings_base(self, value:Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise ValueError("O dicionário de embeddings deve ser do tipo dict.")
        
        if not "list_embeddings" in value:
            raise ValueError("O dicionário de embeddings deve conter a chave 'list_embeddings'.")
        
        if not any(key in value for key in ["list_str", "dict_str"]):
            raise ValueError("O dicionário de embeddings deve conter a chave 'list_str' ou 'dict_str'.")

        
    def __init__(
        self, *,
        model:str,
        host: Literal["http://localhost:11434"]| str | None = None, 
        **kwargs
    ) -> None:    
          
        self.model = model        
        self.__client = ollama.Client(host, **kwargs)
        self.__embeddings_base:Dict[str, Any]| None = None
    
    def generate_list_embeddings_base(self, data_base:List[str]|Dict[str, Any]) -> Dict[str, List[float]|str]:
        embeddings_base = {}
        if isinstance(data_base, dict):
            embeddings_base["dict_str"] = data_base
            embeddings_base["list_embeddings"] = [ollama.embeddings(model=self.model, prompt=str(name)).embedding for name in data_base]
            
            
        elif isinstance(data_base, list):
            embeddings_base["list_str"] = data_base
            embeddings_base["list_embeddings"] = [ollama.embeddings(model=self.model, prompt=str(name)).embedding for name in data_base]
            
        self.__embeddings_base = embeddings_base
        return embeddings_base
    
    def search(self, query:str, *,  list_str_base:List[str]|Dict[str, List[float]]|None=None, accuracy:float=0)  -> tuple:
        if not list_str_base is None:
            self.generate_list_embeddings_base(list_str_base)
        
        if self.embeddings_base is None:
            raise ValueError("A base de embeddings não foi gerada. Por favor, forneça uma lista de strings ou um dicionário com as chaves 'list_str' e 'list_embeddings'.")
        
        q_emb = ollama.embeddings(model=self.model, prompt=str(query)).embedding

        sims = [np.dot(q_emb, d) / (np.linalg.norm(q_emb) * np.linalg.norm(d)) for d in self.embeddings_base["list_embeddings"]]
        
        ranked = sorted(zip((self.embeddings_base["list_str"] if "list_str" in self.embeddings_base else list(self.embeddings_base["dict_str"].keys())), sims), key=lambda x: x[1], reverse=True)
        #import pdb; pdb.set_trace()
        ranked = tuple([value if "list_str" in self.embeddings_base else (self.embeddings_base["dict_str"][value[0]], value[1]) for value in ranked if value[1] >= accuracy])
        
        return ranked
        

        

if __name__ == "__main__":
    bot = Embedding(
        model="snowflake-arctic-embed2:568m",
        host="http://patrimar-rpa2:11434/"
    )
    
    docs = [
    '2300 Rio de Janeiro',
    'Acqua Galleria Condomínio Resort - Condomínio 1',
    'Acqua Galleria Condomínio Resort - Condomínio 2',
    'Acqua Galleria Condomínio Resort - Condomínio 3',
    'Alta Vista Estoril',
    'Atlântico Golf',
    'Brickell',
    'Coral Gables',
    'Duo',
    'Edifício Adelaide Santiago',
    'Edifício Apogée',
    'Edifício Avignon',
    'Edifício Brooklyn',
    'Edifício Gioia Del Colle',
    'Edifício Jornalista Oswaldo Nobre',
    'Edifício José Torres Franco',
    'Edifício Key Biscayne',
    "Edifício L'Essence",
    'Edifício Maura Valadares Gontijo',
    'Edifício Mayfair Offices',
    'Edifício Nashville',
    'Edifício Neuchâtel',
    'Edifício Niagara Falls - Edifício Angel Falls - Edifício Victoria Falls',
    'Edifício Olga Chiari',
    'Edifício Professor Danilo Ambrósio',
    'Edifício Saint Emilion',
    'Edifício Saint Tropez',
    'Edifício Soho Square',
    'Edifício Tribeca Square',
    'Edifício Vivaldi Moreira [Holiday Inn]',
    'Empreendimento Novolar Recreio',
    'Epic Golf Residence',
    'Étoile',
    'Four Seasons Condomínio Resort',
    'Grand Quartier',
    'Grand Quartier 2',
    'Grand Resort Jaraguá',
    'Green View',
    'Greenport Residences',
    'Greenwich Village',
    'High Line Square',
    'Icon Golf Residence',
    'Le Sommet',
    'Madison Square',
    'Manhattan Square',
    'Mia Felicitá Condomínio',
    'Mirante do Jambreiro',
    'Mirante Estoril',
    'Montano Antilia',
    'Novolar Alamedas do Brito',
    'Novolar Absolute',
    'Novolar Atlanta',
    'Novolar Cenarium I',
    'Novolar Flores do Brito',
    'Novolar Green Life',
    'Novolar Jardins',
    'Novolar Jardins do Brito',
    'Novolar Moinho',
    'Novolar Prime View',
    'Novolar Reserva Laguna',
    'Novolar Reserva Pontal',
    'Novolar Sevilha',
    'Novolar Solare',
    'Novolar Valência',
    'Novolar Vargem Grande',
    'Novolar Viena',
    'Oceana Golf II',
    'Palm Springs Pampulha',
    'Palo Alto Residences',
    'Park Residence Condomínio Resort',
    'Priorato Residence',
    'Quintas do Morro',
    'Reserva do Mirataia I',
    'Reserva do Mirataia II',
    'Residencial Inovatto',
    'Residencial Porto Fino',
    'Residencial Ruth Silveira e Ruth Silveira Stores',
    'Residencial Springfield',
    'Residencial Villaggio Novità',
    'Residencial Villaggio Ventura',
    'Skyline',
    'Sunset View',
    'The Plaza',
    'Union Square',
    'Unique',
    'Unique - Avulsos',
    'Villaggio Florença',
    'Villaggio Gutierrez',
    'Villaggio Verona',
    'Vision'
    ]
    # docs = {x:x for x in docs}
    # docs["lessence"] = "Edifício L'Essence"
    # docs["lesence"] = "Edifício L'Essence"
    # docs["Edifício Lessence"] = "Edifício L'Essence"
    
    bot.generate_list_embeddings_base(docs)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in [r"\exit", r"\quit", r"\e"]:
            break
        results = bot.search(user_input)
        print("Results:")
        try:
            for doc, score in results[0:3]:
                print(f"    {doc} (similarity: {score:.4f})")
                #break
        except Exception as err:
            import traceback; print(traceback.format_exc())
            import pdb; pdb.set_trace()
    
    
    