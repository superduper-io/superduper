from superduper.base.datatype import INBUILT_DATATYPES, Vector
from superduper.base.schema import Schema
from superduper.components.listener import Listener
from superduper.components.model import ObjectModel


def test_parse_schema():

    spec = 'a=str|b=vector[float:32]'

    schema = INBUILT_DATATYPES[spec]

    assert schema

    assert isinstance(schema['b'], Vector)


def test_nested(db):

    model = ObjectModel('test', object=lambda x: x, datatype='a=str|b=dillencoder')

    listener = Listener('test', model=model, select=db['test'], key='x')

    table = listener.output_table

    assert isinstance(table.schema[listener.outputs], Schema)

    example = {listener.outputs: {'a': 'hello', 'b': [1, 2, 3]}}

    encoded = table.schema.encode_data(example)

    assert isinstance(encoded[listener.outputs]['b'], str)

    assert encoded[listener.outputs]['b'] == 'gASVCwAAAAAAAABdlChLAUsCSwNlLg=='
