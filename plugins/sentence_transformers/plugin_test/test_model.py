from test.utils.component import utils as component_utils

import sentence_transformers

from superduper_sentence_transformers import SentenceTransformer


def test_encode_and_decode():
    model = SentenceTransformer(
        identifier="embedding",
        object=sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2"),
        postprocess=lambda x: x.tolist(),
        predict_kwargs={"show_progress_bar": True},
    )
    component_utils.test_encode_and_decode(model)
