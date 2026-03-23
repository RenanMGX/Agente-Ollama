from collections.abc import Iterator
from email import message
import os
from pathlib import Path

import ollama
from typing import Any, Callable, Literal, Mapping, Sequence, List, Dict

from ..generate.generate import Generate

from copy import deepcopy

from pydantic.json_schema import JsonSchemaValue


class Chat():
    @property
    def model(self) -> str:
        return self.__model
    
    def __init__(self, model:str, system: str = "", historic_path:Path|str=Path.cwd(), host: Literal["http://localhost:11434"]| str | None = None, **kwargs) -> None:
        if isinstance(historic_path, str):
            historic_path = Path(historic_path)
        if not historic_path.is_dir() or not historic_path.exists():
            raise ValueError(f"historic_path must be a valid directory path. Got: {historic_path}")
        
        
        self.__model = model
        self.__historychat: List[dict] = []
        self.__client = ollama.Client(host, **kwargs)
        self.__system = system
        self.__title = ""
        
        if self.__system:
            self.__historychat.append({"role": "system", "content": system})
            
    def save_historic(self) -> None:
        pass
        
    def restart_chat(self, system: str = "") -> None:
        self.__historychat = []
        if system:
            self.__historychat.append({"role": "system", "content": system})
        
    def chat(self, prompt: str,):
        message = deepcopy(self.__historychat)
        message.append({"role": "user", "content": prompt})
        
        res = self.__client.chat(
            model=self.__model, 
            messages=message,
            stream=False
        )
        if not self.__title:
            self.__title = Generate(model=self.__model).generate(prompt=f"You will receive a prompt. Based on its content, you must generate a title that accurately represents the text. The title must be written in the same language as the original prompt and should be concise, avoiding excessive length.\n prompt:\n{prompt}").response
        
        print(f"Chat Title: {self.__title=}")
        message.append({"role": "assistant", "content": res.message.content})
        #print(message)
        self.__historychat = message
        
        return res
        

if __name__ == "__main__":
    pass
