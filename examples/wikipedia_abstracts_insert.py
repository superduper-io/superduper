import json
import random
import tqdm

from sddb.client import SddbClient

c = SddbClient()
c.drop_database('wikipedia_abstracts')
docs = c.wikipedia_abstracts.documents

with open('data/wikipedia/abstracts.json') as f:
    data = json.load(f)

random.shuffle(data)
data = data[:100000]

docs.create_semantic_index(
    'simple_glove',
    'import',
    {'module': 'mymodels.embeddings', 'class': 'SimpleGlove', 'kwargs': {}}
)

docs.create_imputation(
    'random_classifier',
    'import',
    {'module': 'mymodels.classifiers', 'class': 'RandomClassifier', 'kwargs': {}}
)

tmp = []
for i in tqdm.tqdm(range(len(data))):
    tmp.append(data[i])
    if i > 0 and i % 5000 == 0:
        docs.insert_many(tmp)
        tmp = []
