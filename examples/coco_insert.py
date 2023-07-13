import json
import random
from sddb.client import client as c

with open('data/coco/captions.json') as f:
    captions = json.load(f)
random.shuffle(captions)
captions = captions[:100]
c.drop_database('coco')
docs = c.coco.documents
docs.update_meta_data('valid_probability', 0.05)
docs.update_meta_data('data', 'data/downloads')
docs.update_meta_data('n_download_workers', 0)

docs.create_semantic_index({
    'name': 'average_glove',
    'models': [
        {
            'name': 'average_glove',
            'type': 'import',
            'args': {
                'path': 'mymodels.embeddings.AverageOfGloves',
                'kwargs': {}
            },
            'active': True,
            'converter': 'sddb.models.converters.FloatTensor',
            'key': 'captions',
        }
    ],
    'target': '_base',
})

docs.create_model({
    'name': 'second_download',
    'type': 'import',
    'args': {
        'path': 'mymodels.extractors.Dummy',
        'kwargs': {},
    },
    'key': '_base',
    'active': True,
    'download': True,
    'target': 'second_download'
})

docs.create_model({
    'name': 'second_feature',
    'type': 'import',
    'args': {
        'path': 'mymodels.embeddings.Identity',
        'kwargs': {},
    },
    'key': '_base',
    'features': {'captions': 'average_glove'},
    'active': True,
    'converter': 'sddb.models.converters.FloatTensor',
    'dependencies': ["average_glove"]
})

docs.insert_many(captions, verbose=True)