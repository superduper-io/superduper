import dataclasses as dc
import functools
import os
import typing as t

import requests
from llama_cpp import Llama

from superduperdb.components.model import Model


def download_uri(uri, save_path):
    response = requests.get(uri)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Error while downloading uri {uri}")


@dc.dataclass
class LlamaCpp(Model):
    model_name_or_path: str = "facebook/opt-125m"
    object: t.Optional[Llama] = None
    model_kwargs: t.Dict = dc.field(default_factory=dict)
    download_dir: str = '.llama_cpp'

    def __post_init__(self):
        if self.model_name_or_path.startswith('http'):
            # Download the uri
            os.makedirs(self.download_dir, exist_ok=True)
            saved_path = os.path.join(self.download_dir, f'{self.identifier}.gguf')

            download_uri(self.model_name_or_path, saved_path)
            self.model_name_or_path = saved_path

        if self.predict_kwargs is None:
            self.predict_kwargs = {}

        self._model = Llama(self.model_name_or_path, **self.model_kwargs)
        super().__post_init__()

    def _predict(
        self,
        X: t.Union[str, t.List[str], t.List[dict[str, str]]],
        one: bool = False,
        **kwargs: t.Any,
    ):
        one = isinstance(X, str)

        assert isinstance(self.predict_kwargs, dict)
        to_call = functools.partial(
            self._model.create_completion, **self.predict_kwargs
        )
        if one:
            return to_call(X)
        else:
            return list(map(to_call, X))


@dc.dataclass
class LlamaCppEmbedding(LlamaCpp):
    def __post_init__(self):
        self.model_kwargs['embedding'] = True
        super().__post_init__()

    def _predict(
        self,
        X: t.Union[str, t.List[str], t.List[dict[str, str]]],
        one: bool = False,
        **kwargs: t.Any,
    ):
        one = isinstance(X, str)
        assert isinstance(self.predict_kwargs, dict)

        to_call = functools.partial(self._model.create_embedding, **self.predict_kwargs)
        if one:
            return to_call(X)
        else:
            return list(map(to_call, X))
