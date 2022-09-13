import json
from sddb.client import client as c

if __name__ == '__main__':

    with open('data/ebay/data.json') as f:
        data = json.load(f)

    data = data[:10]

    if 0:
        c.drop_database('ebay')
        docs = c.ebay.documents
        docs.update_meta_data('data', 'data/downloads')
        docs.download_timeout = 3

        docs.create_model({
            'name': 'parse',
            'type': 'import',
            'args': {
                'path': 'mymodels.extractors.Parser',
                'kwargs': {},
            },
            'key': 'page',
            'active': True,
            'download': True,
            'target': 'information',
            'requires': 'page'
        })

        docs.insert_many(data, verbose=True)
    else:
        docs = c.ebay.documents
