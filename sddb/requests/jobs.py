from sddb import cf
import requests


def download_content(database, collection, ids, dependencies=()):
    print('submitting:')
    print(f'''
        download_content(
            {database},
            {collection},
            {ids},
            dependencies={dependencies},
        )
    ''')
    ids = [str(id_) for id_ in ids]
    r = requests.post(
        f'http://{cf["jobs"]["host"]}:{cf["jobs"]["port"]}/download_content',
        json={
            'database': database,
            'collection': collection,
            'ids': ids,
            'dependencies': list(dependencies),
        }
    )
    return r.text


def process_documents_with_model(
    database,
    collection,
    model_name,
    ids,
    batch_size=10,
    verbose=False,
    blocking=False,
    dependencies=(),
):
    print('submitting:')
    print(f'''
        process_documents_with_model(
            {database},
            {collection},
            {model_name},
            {ids},
            batch_size={batch_size},
            verbose={verbose},
            blocking={blocking},
            dependencies={dependencies},
        )
    ''')
    ids = [str(id_) for id_ in ids]
    r = requests.post(
        f'http://{cf["jobs"]["host"]}:{cf["jobs"]["port"]}/process_documents_with_model',
        json={
            'collection': collection,
            'database': database,
            'model_name': model_name,
            'batch_size': batch_size,
            'verbose': verbose,
            'ids': ids,
            'blocking': blocking,
            'dependencies': list(dependencies),
        }
    )
    return r.text
