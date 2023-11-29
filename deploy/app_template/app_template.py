import dotenv
from fastapi import FastAPI
from superduperdb import superduper
from superduperdb import Document
from superduperdb.backends.mongodb import Collection

dotenv.load_dotenv('<path-to-dotenv-file>')
db = superduper('<data-backend-uri>')
collection = Collection('<collection-name>')


app = FastAPI()


@app.get("/")
def show():
    return {"models": db.show('model'), 'listeners': db.show('listener'), 'vector_indexes': db.show('vector_index')}


@app.get("/search")
def search(input: str):
    results = sorted(list(
        collection
            .like(Document({'<key>': input}), vector_index='<index-name>', n=20)
            .find({}, {'_id': 0}),
        key=lambda x: -x['score'],
    ))
    return {'results': results}


@app.get("/predict")
def predict(input: str):
    num_results = 5
    output, _ = db.predict(
        model_name='<model-name>',
        input=input,
        context_select=(
            collection
                .like(Document({'<key>': input}), vector_index='<index-name>', n=num_results)
                .find()
        ),
        context_key='txt',
    )
    return {'prediction': output}