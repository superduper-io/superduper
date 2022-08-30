import json
import random
from sddb.client import SddbClient

with open('data/coco/captions.json') as f:
    captions = json.load(f)
random.shuffle(captions)
captions = captions[:100]
for r in captions:
    del r['image']['_content']['path']

c = SddbClient()
c.drop_database('coco')
docs = c.coco.documents
docs.update_meta_data('valid_probability', 0.05)
docs.update_meta_data('data', 'data/downloads')
docs.update_meta_data('num_download_workers', 20)

docs.create_semantic_index({
    'name': 'average_glove',
    'models': {
        'captions': {
            'name': 'average_glove',
            'type': 'import',
            'args': {
                'path': 'mymodels.embeddings.AverageOfGloves',
                'kwargs': {}
            },
            'active': True,
            'converter': 'sddb.converters.FloatTensor',
            'key': 'captions'
        }
    },
    'target': 'captions'
})

docs.insert_many(captions, verbose=True)