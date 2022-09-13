from bson import BSON
import requests

from sddb import cf


def find(database, collection, *args, **kwargs):
    s = requests.Session()
    with s.get(f'http://{cf["master"]["host"]}:{cf["master"]["port"]}/find',
               json={
                   'database': database,
                   'collection': collection,
                   'args': args,
                   'kwargs': kwargs,
               },
               headers=None,
               stream=True) as response:
        for line in response.iter_lines():
            if line:
                print(line)


def find_one(database, collection, filter, *args, **kwargs):
    response = requests.get(
        f'http://{cf["master"]["host"]}:{cf["master"]["port"]}/find_one',
        json={
           'database': database,
           'collection': collection,
           'filter': bytes(BSON.encode(filter)).decode('iso-8859-1'),
           'args': list(args),
           'kwargs': kwargs,
        },
        headers=None,
        stream=True
    )
    r = BSON.decode(response.content)
    from sddb.collection import SddbCursor
    return SddbCursor.convert_types(r, convert=True)
