import base64
import json
from copy import deepcopy

from superduper.base.document import Document
from superduper.components.component import Component
from superduper.misc.special_dicts import recursive_update


def _bytes_to_base64(obj):
    def replace_func(value):
        if isinstance(value, bytes):
            return base64.b64encode(value).decode()
        return value

    return recursive_update(obj, replace_func)


def test_encode_and_decode(component: Component):
    encode_data = component.encode()

    # Make sure that the data is JSON serializable
    assert json.dumps(_bytes_to_base64(deepcopy(encode_data)))

    # Make sure that the data can be decoded to the same component
    load_component = Component.decode(encode_data)
    load_component.setup()

    assert type(load_component) is type(component)
    assert load_component.metadata == component.metadata

    return load_component
