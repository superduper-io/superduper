import ibis
import pandas
import PIL.Image
import pytest
from sklearn import svm
import random
import torchvision

from superduperdb import superduper
from superduperdb.db.ibis.schema import IbisSchema
from superduperdb.ext.pillow.image import pil_image
from superduperdb.ext.torch.model import TorchModel
from superduperdb.ext.sklearn.model import Estimator, SklearnTrainingConfiguration


chars = 'abcdefghijklmnopqrrstuvwxyz'


def random_word(n_char):
    return ''.join([random.choice(list(chars)) for _ in range(n_char)])


def random_str(n_char, n_words):
    return ' '.join([random_word(n_char) for _ in range(n_words)])


def resnet():
    return torchvision.models.resnet18()


@pytest.fixture
def product_data():
    connection = ibis.sqlite.connect("mydb.sqlite")

    db = superduper(connection)

    db.add(
        IbisSchema(
            table_name='products',
            schema={
                'id': 'int64',
                'brand': 'string',
                'name': 'string',
                'image': pil_image,
            }
        )
    )

    db.add(
        IbisSchema(
            table_name='users',
            schema={
                'id': 'int64',
            }
        )
    )

    db.add(
        IbisSchema(
            table_name='user_journeys',
            schema={
                'id': 'int64',
                'user_id': 'int64',
                'product_id': 'int64',
                'rating': 'float',
                'gender': 'float',
            }
        )
    )

    image = PIL.Image.open('test/material/data/test-image.jpg')

    sample_products = pandas.DataFrame([
        {
            'brand': f'Nike {random_word(3)}',
            'name': random_str(4, 3), 
            'image': image
        }
        for _ in range(10)
    ])

    table = db['products']
    db.execute(table.insert(sample_products))

    sample_users = pandas.DataFrame([
        {'gender': ['M', 'F'][round(random.random())]}
        for _ in range(10)
    ])
    table = db['users']
    db.execute(table.insert(sample_users))

    user_product_clicks = pandas.DataFrame(sum([
        [{'user_id': i, 'product_id': random.randrange(10)} for _ in range(2)]
        for i in range(10)
    ], []))
    table = db['user_product_clicks']
    db.execute(table.insert(user_product_clicks))

    yield db

    db.drop(force=True)


def test_predict_with_model(product_data, resnet):

    # Basic setup to predict whether a product is better suited to women or men
    # based on the product image

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

    def estimator_preprocess(r):
        # TODO Don't know what the convention will be for the naming of outputs
        return r['_outputs.product__image.product_featurizer']

    product_model = TorchModel(
        identifier='product_featurizer',
        object=resnet,
        preprocess=preprocess,
    )

    products = product_data['products']
    users = product_data['users']
    journeys = product_data['user_journeys']

    product_model.predict(
        X='image',
        db=product_data,
        select=products.all(),
    )

    product_outputs = product_data.outputs['product_model']

    q = (
        users.join(
            journeys,
            journeys.user_id == users.id
        ).join(
            products,
            journeys.product_id == products.id
        ).join(
            product_outputs.input_id == users.id
            & product_outputs.table_name == users.name,
        )
    )

    o = Estimator(
        identifier='gender_estimator',
        object=svm.SVC(),
        preprocess=estimator_preprocess,
    )

    def y_preprocess(gender):
        return {'M': 0, 'F': 1}[gender]

    o.fit(
        X='_base',
        y='gender',
        select=q,
        db=product_data,
        configuration=SklearnTrainingConfiguration(
            identifier='skconfig',
            y_preprocess=y_preprocess,
        ),
    )
    o.predict(X='_base', db=product_data, select=users.all())