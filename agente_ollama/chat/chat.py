import os
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

from ollama._types import (
  ChatRequest,
  ChatResponse,
  CopyRequest,
  CreateRequest,
  DeleteRequest,
  EmbeddingsRequest,
  EmbeddingsResponse,
  EmbedRequest,
  EmbedResponse,
  GenerateRequest,
  GenerateResponse,
  Image,
  ListResponse,
  Message,
  Options,
  ProcessResponse,
  ProgressResponse,
  PullRequest,
  PushRequest,
  ResponseError,
  ShowRequest,
  ShowResponse,
  StatusResponse,
  Tool,
  WebFetchRequest,
  WebFetchResponse,
  WebSearchRequest,
  WebSearchResponse,
)


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
    @property
    def model(self) -> str:
        return self.__model
    
    @property
    def title(self) -> str:
        return self.__title
    
    def __init__(self, model:str, system: str = "", host: Literal["http://localhost:11434"]| str | None = None, **kwargs) -> None:      
        self.__model = model
        self.__historychat: List[dict] = []
        self.__client = ollama.Client(host, **kwargs)
        self.__system = system
        self.__title = ""
        self.save:bool = True
        self.register_image_in_historic = False
        
        if self.__system:
            self.__historychat.append({"role": "system", "content": system})
            
        self.__historic_manager = HistoricManager()
            
        
    def restart_chat(self, system: str = "") -> None:
        self.__historychat = []
        if system:
            self.__historychat.append({"role": "system", "content": system})
        
    def chat(self, 
            prompt: str,
            title:str="",
            format:Literal['', 'json']='',
            images_paths:List[str]=[],
            temperature:float|None=None,
            top_p:float|None=None,
            seed:int|None=None,
            num_predictions:int|None=None,
            think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
            
            logprobs: Optional[bool] = None,
            top_logprobs: Optional[int] = None,
            keep_alive: Optional[Union[float, str]] = None,
            tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
            
        ) -> ChatResponse:
        
        options = {}
        if temperature is not None:
            options['temperature'] = temperature
        if top_p is not None:
            options['top_p'] = top_p
        if seed is not None:
            options['seed'] = seed
        if num_predictions is not None:
            options['num_predictions'] = num_predictions
        
        
        _images = []
        if images_paths:
            for image_path in images_paths:
                path = Path(image_path)
                if not path.is_file() or not path.exists():
                    raise ValueError(f"Image path '{image_path}' is not a valid file.")
                if not path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                    raise ValueError(f"Image path '{image_path}' is not a supported image format.")
                
                _images.append(base64.b64encode(path.read_bytes()).decode("utf-8"))
        
        if title:
            historic = self.__historic_manager.get_chat(title)
            if historic:
                self.__historychat = historic
            
        message = deepcopy(self.__historychat)
        
        role:dict = {"role": "user", "content": prompt}
        if _images:
            role['images'] = _images
        if options:
            role['options'] = options
        
        message.append(role)
        
        ##########     CHAT
        res = self.__client.chat(
            model=self.__model, 
            messages=message,
            stream=False,
            format=format,
            think=think,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            keep_alive=keep_alive,
            tools=tools
        )
        ##########
        
        if not title:
            if not self.__title:
                if self.save:
                    self.__title = Generate(model=self.__model).generate(prompt=f"You will receive a prompt. Based on its content, you must generate a title that accurately represents the text. The title must be written in the same language as the original prompt and should be concise, avoiding excessive length.\n prompt:\n{prompt}").response
        else:
            self.__title = title
        
        #print(f"Chat Title: {self.__title=}")
        message.append({"role": "assistant", "content": res.message.content})
        #print(message)
        self.__historychat = message
        if self.save:
            if not self.register_image_in_historic:
                for msg in self.__historychat:
                    if "images" in msg:
                        del msg["images"]
            self.__historic_manager.register_chat(self.__title, self.__historychat)
        
        
        return res
    
    def chat_tools(
        self, 
        prompt:str, 
        tools:List[Any],
        think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
        system:Literal["<DEFAULT_SYSTEM>"]|str = ""
    ):
        tools_message = []
        if system == "<DEFAULT_SYSTEM>":
            if self.__system:
                tools_message.append({"role": "system", "content": self.__system})
        else:
            if system:
                tools_message.append({"role": "system", "content": system})
        
        tools_message.append({"role": "user", "content": prompt})
        
        #for _ in range(len())
        ##########     CHAT
        res = self.__client.chat(
            model=self.__model, 
            messages=tools_message,
            stream=False,
            think=think,
            tools=tools
        )
        ##########
        
        #import pdb;pdb.set_trace()
        tools_map = {t.__name__: t for t in tools if callable(t)}
        
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
            model=self.__model, 
            messages=tools_message,
            stream=False,
            think=think,
        )
        
        tools_message.append({"role": "assistant", "content": res.message.content})
        
        print(f"\n\n{tools_message}\n\n")
        return res
        
        

if __name__ == "__main__":
    pass
