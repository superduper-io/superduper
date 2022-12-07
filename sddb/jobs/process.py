import contextlib
import datetime
import traceback

from sddb.client import SddbClient
from sddb import cf


def _handle_function_output(func, database, collection, *args, identifier=None, **kwargs):
    path = f'logs/job-{identifier}.out'
    try:
        with open(path, 'w') as f:
            with contextlib.redirect_stdout(f):
                with contextlib.redirect_stderr(f):
                    collection = SddbClient(**cf['mongodb'])[database][collection]
                    collection.single_thread = True
                    return func(collection, *args, **kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        with open(path, 'a') as f:
            with contextlib.redirect_stdout(f):
                with contextlib.redirect_stderr(f):
                    print(tb)
        collection['_errors'].insert_one({
            'identifier': identifier,
            'time': datetime.datetime.now(),
            'msg': tb,
        })
        raise e


def _download_content(collection, ids):
    return collection._download_content(ids=ids)


def download_content(database, collection, ids, identifier=None):
    return _handle_function_output(
        _download_content,
        database,
        collection,
        ids,
        identifier=identifier,
    )


def _process_documents_with_model(collection, model_name, ids, batch_size=10, verbose=False):
    collection._process_documents_with_model(
        model_name=model_name,
        ids=ids,
        batch_size=batch_size,
        verbose=verbose
    )


def process_documents_with_model(database, collection, model_name, ids, batch_size=10,
                                 verbose=False, identifier=None):
    return _handle_function_output(
        _process_documents_with_model,
        database,
        collection,
        model_name,
        ids,
        batch_size=batch_size,
        identifier=identifier,
        verbose=verbose,
    )
