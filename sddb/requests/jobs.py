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


def train_imputation(database, collection, name):
    print('submitting:')
    print(f'''
        train_imputation(
            {database},
            {collection},
            {name},
        )
    ''')
    r = requests.post(
        f'http://{cf["jobs"]["host"]}:{cf["jobs"]["port"]}/train_imputation',
        json={
            'collection': collection,
            'database': database,
            'name': name,
        }
    )
    return r.text


def train_semantic_index(database, collection, name):
    print('submitting:')
    print(f'''
        train_semantic_index(
            {database},
            {collection},
            {name},
        )
    ''')
    r = requests.post(
        f'http://{cf["jobs"]["host"]}:{cf["jobs"]["port"]}/train_semantic_index',
        json={
            'collection': collection,
            'database': database,
            'name': name,
        }
    )
    return r.text


def process(database, collection, method, *args, dependencies=(), **kwargs):
    if 'ids' in kwargs:
        kwargs['ids'] = [str(id_) for id_ in kwargs['ids']]
    r = requests.post(
        f'http://{cf["jobs"]["host"]}:{cf["jobs"]["port"]}/process_documents_with_model',
        json={
            'collection': collection,
            'database': database,
            'method': method,
            'args': list(args),
            'kwargs': kwargs,
            'dependencies': list(dependencies),
        }
    )
    if r.status_code == 500:
        raise Exception(r.text)
    else:
        return r.text



def process_documents_with_model(
    database,
    collection,
    model_name,
    ids,
    dependencies=(),
    **kwargs,
):
    print('submitting:')
    print(f'''
        process_documents_with_model(
            {database},
            {collection},
            {model_name},
            {ids},
            kwargs={kwargs},
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
            'ids': ids,
            'kwargs': kwargs,
            'dependencies': list(dependencies),
        }
    )
    if r.status_code == 500:
        raise Exception(r.text)
    else:
        return r.text
