import dataclasses as dc
import os
import typing as t

import requests
from llama_cpp import Llama
from superduper.components.llm.model import BaseLLM


# TODO use core downloader already implemented
def download_uri(uri, save_path):
    """Download file.

    :param uri: URI to download
    :param save_path: place to save
    """
    response = requests.get(uri)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Error while downloading uri {uri}")


class LlamaCpp(BaseLLM):
    """Llama.cpp connector.

    :param model_name_or_path: path or name of model
    :param model_kwargs: dictionary of init-kwargs
    :param download_dir: local caching directory

    Example:
    -------
    >>> from superduper_llama_cpp.model import LlamaCpp
    >>>
    >>> model = LlamaCpp(
    >>>     identifier="llm",
    >>>     model_name_or_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    >>> )
    >>> model.predict("Hello world")

    """

    model_name_or_path: str = "facebook/opt-125m"
    model_kwargs: t.Dict = dc.field(default_factory=dict)
    download_dir: str = '.llama_cpp'
    signature: str = 'singleton'

    def setup(self):
        """Initialize the model.

        If the model_name_or_path is a uri, download it to the download_dir.
        """
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
        """Generate text from a prompt.

        :param prompt: The prompt to generate text from.
        :param kwargs: The keyword arguments to pass to the llm model.
        """
        out = self._model.create_completion(prompt, **self.predict_kwargs, **kwargs)
        return out['choices'][0]['text']


class LlamaCppEmbedding(LlamaCpp):
    """Llama.cpp connector for embeddings."""

    def _generate(self, prompt: str, **kwargs) -> str:
        """Generate embedding from a prompt.

        :param prompt: The prompt to generate the embedding from.
        :param kwargs: The keyword arguments to pass to the llm model.
        """
        return self._model.create_embedding(
            prompt, embedding=True, **self.predict_kwargs, **kwargs
        )
