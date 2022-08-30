from sddb import cf
import requests


def process_documents_with_model(
    database,
    collection,
    model_name,
    ids,
    batch_size=10,
    verbose=False
):
    requests.get(
        f'{cf["jobs"]["url"]}/process_documents_with_model',
        params={
            'collection': collection,
            'database': database,
            'model_name': model_name,
            'batch_size': batch_size,
            'verbose': verbose,
            'ids': ids,
        }
    )
