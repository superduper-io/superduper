from sddb import cf
import requests


def process_documents_with_model(
    database,
    collection,
    model_name,
    ids,
    batch_size=10,
    verbose=False,
    blocking=False,
):
    ids = [str(id_) for id_ in ids]
    requests.post(
        f'http://{cf["jobs"]["host"]}:{cf["jobs"]["port"]}/process_documents_with_model',
        json={
            'collection': collection,
            'database': database,
            'model_name': model_name,
            'batch_size': batch_size,
            'verbose': verbose,
            'ids': ids,
            'blocking': blocking,
        }
    )
