import os
import re
import sys
from turtle import pd
from httpx import delete
import ollama
import json
import base64

from collections.abc import Iterator
from email import message
from pathlib import Path

from typing import Any, Callable, Literal, Mapping, Sequence, List, Dict, Union, Optional
from ..generate.generate import Generate
from copy import deepcopy
from pydantic.json_schema import JsonSchemaValue
from datetime import date, datetime

from ollama._types import (
  ChatResponse,
  Tool,
)

from ddgs import DDGS

def buscar_web(query: str, max_results: int=10 ) -> str:
    """Realiza uma busca na web usando DuckDuckGo e retorna os resultados como uma string.
    Args:
        query (str): A consulta de busca.
        max_results (int): O número máximo de resultados a serem retornados.
    """
    for _ in range(3):
        with DDGS() as ddgs:
            resultados = list(ddgs.text(query, max_results=max_results))
        if resultados:
            return str(resultados)
    return "No search results found."


class HistoricManager():
    @property
    def data(self) -> Dict[str, Any]:
        return self.__data
    
    def __init__(self, path:Path|str=Path.cwd().joinpath("historic", "chat.json")):
        if isinstance(path, str):
            path = Path(path)
        
        if path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.name.endswith(".json"):
                path = path.with_suffix(".json")
            
        elif path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            path = path.joinpath("chat.json")
            
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
                
        self.__data: Dict[str, Any] = {}
        self.__path = path
                
    def __save(self, data: dict):
        with open(self.__path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
    def __load(self) -> dict:
        with open(self.__path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.__data = data
        return data
    
    def register_chat(self, title: str, chat: List[dict]):
        data = self.__load()
        data[title] = chat
        self.__save(data)
        
    def list_chats(self) -> List[str]:
        data = self.__load()
        return list(data.keys())
    
    def get_chat(self, title: str) -> List[dict]:
        data = self.__load()
        return data.get(title, [])
    
    def delete_chat(self, title: str):
        data = self.__load()
        if title in data:
            del data[title]
            self.__save(data)
    
    def clear_all_chats(self):
        self.__save({})

class Chat():
    # @property
    # def model(self) -> str:
    #     return self.__model
        
    def __init__(
        self, 
        model:str, 
        chat_sistem:str = "",
        host: Literal["http://localhost:11434"]| str | None = None, 
        **kwargs
    ) -> None:    
          
        self.model = model
        
        self.historicchat: List[dict] = []
        
        self.__client = ollama.Client(host, **kwargs)

        self.chat_sistem = chat_sistem
        # if self.__system:
        #     self.__historychat.append({"role": "system", "content": system})
            
            
        
    def restart_chat(self, system: str = "") -> None:
        self.historicchat = []
        if system:
            self.historicchat.append({"role": "system", "content": system})
            
    @staticmethod
    def convert_images_to_base64(images_paths: List[str]) -> List[dict]:
        images = []
        for image_path in images_paths:
            path = Path(image_path)
            if not path.is_file() or not path.exists():
                raise ValueError(f"Image path '{image_path}' is not a valid file.")
            # if not path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".pdf"]:
            #     raise ValueError(f"Image path '{image_path}' is not a supported image format.")
            
            images.append(base64.b64encode(path.read_bytes()).decode("utf-8"))
        return images
        
    def chat(self, 
            prompt: str,
            web_search: Optional[bool],
            system:str="",
            format:Literal['', 'json']='',
            images_paths:List[str]=[],
            temperature:float|None=None,
            top_p:float|None=None,
            seed:int|None=None,
            num_predictions:int|None=None,
            think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
            date:datetime|None = None,
            remove_historic:bool=False,
            
            logprobs: Optional[bool] = None,
            top_logprobs: Optional[int] = None,
            keep_alive: Optional[Union[float, str]] = None,
            #tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
            
        ) -> ChatResponse:
        if isinstance(date, datetime):
            _date = date
        else:
            _date = datetime.now()
        
        options = {}
        if temperature is not None:
            options['temperature'] = temperature
        if top_p is not None:
            options['top_p'] = top_p
        if seed is not None:
            options['seed'] = seed
        if num_predictions is not None:
            options['num_predictions'] = num_predictions
        
        
        _images = self.convert_images_to_base64(images_paths) if images_paths else []
        
        #create historic   
        if remove_historic:       
            message = []
        else:
            message = deepcopy(self.historicchat)
            
        #system unifique
        system_temp = [self.chat_sistem] if self.chat_sistem else []
        for msg in message:
            if msg['role'] == 'system':
                sys_temp = msg['content'].strip().split("\n")
                for s in sys_temp:
                    system_temp.append(s)
                message.remove(msg)
        
        if not system in system_temp:
            system_temp.append(system)
            
        system_temp = list(set(system_temp))
        system_temp = [s for s in system_temp if s.strip()]
                    
        message.insert(0,{"role": "system", "content": ("\n".join(system_temp)).strip()})
        
        #import pdb;pdb.set_trace()
         
        
        message.append({ "role": "assistant", "tool_calls": [ { 'type': 'function', 'function': { 'name': "datetime_now", 'arguments': {} }, } ] } )
        message.append({"role": "tool", "tool_name": "datetime_now", "content": _date.strftime("%Y-%m-%d %H:%M:%S")})
        
            
        
        role:dict = {"role": "user", "content": prompt}
        if _images:
            role['images'] = _images
        
        message.append(role)
        
        if web_search:
            try:
                #web_search
                search = self.web_search(
                    query=prompt
                )
                if search:
                    for item in search:
                        message.append(item)
            except Exception as err:
                pass
                #print(f"Error during web search: {err}")
                #message.append({"role": "assistant", "content": buscar_web(prompt)})
                #message.append({ "role": "assistant", "tool_calls": [ { 'type': 'function', 'function': { 'name': #"buscar_web", 'arguments': {"query": prompt} }, } ] } )
                #message.append({"role": "tool", "tool_name": "buscar_web", "content": buscar_web(prompt)})
        #print(f"\n\n{message=}\n\n")
        
        ##########     CHAT
        res = self.__client.chat(
            model=self.model, 
            messages=message,
            stream=False,
            format=format,
            think=think,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            keep_alive=keep_alive,
            options=options if options else None,
            #tools=tools
        )
        ##########
        
        # if not title:
        #     if not self.__title:
        #         if self.save:
        #             self.__title = Generate(model=self.__model).generate(prompt=f"You will receive a prompt. Based on its content, you must generate a title that accurately represents the text. The title must be written in the same language as the original prompt and should be concise, avoiding excessive length.\n prompt:\n{prompt}").response
        # else:
        #     self.__title = title
        
        #print(f"Chat Title: {self.__title=}")
        if res.message.content:
            message.append({"role": "assistant", "content": res.message.content})
        else:
            raise Exception(f"{res}")
        
        _message = []
        for msg in message:
            if msg['role'] == 'tool':
                #message.remove(msg)
                continue
            if 'tool_calls' in msg:
                #message.remove(msg)
                continue
            #import pdb;pdb.set_trace()
            _message.append(msg)
                
        #print(message)
        if not remove_historic:
            self.historicchat = _message
        
        return res
    
    def web_search(self, query:str) -> List:
        tools_message = []
        tools_message.append({"role": "system", "content": "You are a helpful assistant with access to a web search tool. When the user provides a query, you will use the 'buscar_web' tool to search the web and return relevant information."})
        
        tools_message.append({"role": "user", "content": query})
        
        tools = [buscar_web]
        tools_map = {t.__name__: t for t in tools if callable(t)}
        res = self.__client.chat(
            model=self.model, 
            messages=tools_message,
            stream=False,
            tools=tools,
        )
        ##########
        
        #import pdb;pdb.set_trace()
        tools_return = []
        if res.message.tool_calls:
            for tool in res.message.tool_calls:
                function = tool.function.name
                args = tool.function.arguments
                                        
                # procura a função nas tools passadas, senão em globals()
                func = tools_map.get(function) or globals().get(function)
                if func is None:
                    raise NameError(f"Tool function '{function}' não encontrada")

                # chamar adequadamente conforme tipo de args
                if isinstance(args, dict):
                    result_function = func(**args)
                elif isinstance(args, (list, tuple)):
                    result_function = func(*args)
                else:
                    result_function = func(args) # type: ignore
                
                # tools_message.append({"role": "assistant", "content": result_function})
                tools_return.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                'type': 'function',
                                'function': {
                                    'name': function,
                                    'arguments': args
                                },
                            }
                        ]
                    }
                )
                
                tools_return.append({"role": "tool", "tool_name": function, "content": result_function})
                        
        return tools_return
        
    def functions_tools(
        self, 
        prompt:str,
        tools:List[Any],
        think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
        system:str = "",
    ) -> ChatResponse:
        tools_message = []
        if system:
            tools_message.append({"role": "system", "content": system})
        
        tools_message.append({"role": "user", "content": prompt})
        
        #for _ in range(len())
        ##########     CHAT
        tools_map = {t.__name__: t for t in tools if callable(t)}
        res = self.__client.chat(
            model=self.model, 
            messages=tools_message,
            stream=False,
            think=think,
            tools=tools,
            
        )
        ##########
        
        #import pdb;pdb.set_trace()
        
        if res.message.tool_calls:
            for tool in res.message.tool_calls:
                function = tool.function.name
                args = tool.function.arguments
                                
                
                # procura a função nas tools passadas, senão em globals()
                func = tools_map.get(function) or globals().get(function)
                if func is None:
                    raise NameError(f"Tool function '{function}' não encontrada")

                # chamar adequadamente conforme tipo de args
                if isinstance(args, dict):
                    result_function = func(**args)
                elif isinstance(args, (list, tuple)):
                    result_function = func(*args)
                else:
                    result_function = func(args)                
                
                # tools_message.append({"role": "assistant", "content": result_function})
                tools_message.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                'type': 'function',
                                'function': {
                                    'name': function,
                                    'arguments': args
                                },
                            }
                        ]
                    }
                )
                
                
                tools_message.append({"role": "tool", "tool_name": function, "content": result_function})
                
                
        res = self.__client.chat(
            model=self.model, 
            messages=tools_message,
            stream=False,
            think=think,
        )
        
        tools_message.append({"role": "assistant", "content": res.message.content})
        
        print(f"\n\n{tools_message}\n\n")
        return res
        
        

if __name__ == "__main__":
    pass
