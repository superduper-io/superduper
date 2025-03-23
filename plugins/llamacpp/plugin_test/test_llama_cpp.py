from superduper_llamacpp.model import LlamaCpp, LlamaCppEmbedding


class _MockedLlama:
    def create_completion(self, *args, **kwargs):
        return {'choices': [{'text': 'tested'}]}

    def create_embedding(self, *args, **kwargs):
        return [1]


def test_llama():
    def mocked_init(self):
        self._model = _MockedLlama()
        self.predict_kwargs = {}

    LlamaCpp.setup = mocked_init

    llama = LlamaCpp(
        identifier='myllama',
        model_name_or_path='some_model',
        model_kwargs={'vocab_only': True},
    )

    text = 'testing prompt'
    output = llama.predict(text)
    assert output == 'tested'


def test_llama_embedding():
    def mocked_init(self):
        self._model = _MockedLlama()
        self.predict_kwargs = {}

    LlamaCppEmbedding.init = mocked_init

    llama = LlamaCppEmbedding(
        identifier='myllama',
        model_name_or_path='some_model',
        model_kwargs={'vocab_only': True},
    )

    text = 'testing prompt'
    output = llama.predict(text)
    assert output == [1]
