import dataclasses as dc
import os
import typing as t

import requests
from llama_cpp import Llama

from superduperdb.ext.llm.base import _BaseLLM


# TODO use core downloader already implemented
def download_uri(uri, save_path):
    """
    Download file

    :param uri: URI to download
    :param save_path: place to save
    """
    response = requests.get(uri)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Error while downloading uri {uri}")


@dc.dataclass
class LlamaCpp(_BaseLLM):
    """
    Llama.cpp connector

    :param model_name_or_path: path or name of model
    :param model_kwargs: dictionary of init-kwargs
    :param download_dir: local caching directory
    :param signature: s
    """

    signature: t.ClassVar[str] = 'singleton'

    model_name_or_path: str = "facebook/opt-125m"
    model_kwargs: t.Dict = dc.field(default_factory=dict)
    download_dir: str = '.llama_cpp'

    def init(self):
        if self.model_name_or_path.startswith('http'):
            # Download the uri
            os.makedirs(self.download_dir, exist_ok=True)
            saved_path = os.path.join(self.download_dir, f'{self.identifier}.gguf')

            download_uri(self.model_name_or_path, saved_path)
            self.model_name_or_path = saved_path

        if self.predict_kwargs is None:
            self.predict_kwargs = {}

        self._model = Llama(self.model_name_or_path, **self.model_kwargs)

    def _generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.
        """
        return self._model.create_completion(prompt, **self.predict_kwargs, **kwargs)


@dc.dataclass
class LlamaCppEmbedding(LlamaCpp):
    def _generate(self, prompt: str, **kwargs) -> str:
        """
        Generate embedding from a prompt.
        """
        return self._model.create_embedding(
            prompt, embedding=True, **self.predict_kwargs, **kwargs
        )
