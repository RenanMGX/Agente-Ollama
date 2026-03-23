from collections.abc import Iterator
from typing import Any, Literal, Mapping, Sequence

import ollama
from pydantic.json_schema import JsonSchemaValue
from ..model_manager.models import Models

from ollama._types import (
  GenerateResponse,
  Image,
)

class Generate(ollama.Client):
    @property
    def model(self) -> str:
        return self.__model
    
    def __init__(self, model:str, host: Literal["http://localhost:11434"]| str | None = None, **kwargs) -> None:
        self.__model = model
            
        super().__init__(host, **kwargs)
        
    def generate(self, prompt: str, model:str|None=None, suffix: str | None = None, *, system: str | None = None, template: str | None = None, context: Sequence[int] | None = None, think: bool | None = None, logprobs: bool | None = None, top_logprobs: int | None = None, raw: bool | None = None, format: dict[str, Any] | None | Literal[''] | Literal['json'] = None, images: Sequence[str | bytes | Image] | None = None, options: Mapping[str, Any] | ollama.Options | None = None, keep_alive: float | str | None = None) -> ollama.GenerateResponse:#type: ignore
        return super().generate(
            model=self.__model if model is None else model, #type: ignore
            prompt=prompt, 
            suffix=suffix, #type: ignore
            system=system, #type: ignore
            template=template, #type: ignore
            context=context, 
            stream=False, #type: ignore
            think=think, 
            logprobs=logprobs, 
            top_logprobs=top_logprobs, 
            raw=raw, #type: ignore
            format=format, 
            images=images, 
            options=options, 
            keep_alive=keep_alive)
    
    
if __name__ == "__main__":
    pass