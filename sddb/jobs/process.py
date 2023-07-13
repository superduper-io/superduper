from sddb.client import SddbClient
from sddb import cf


def process_documents_with_model(
    database,
    collection,
    model_name,
    ids,
    batch_size=10,
    verbose=False
):
    print(database)
    print(cf)
    print(collection)
    SddbClient(**cf['mongodb']).list_database_names()
    collection = SddbClient(**cf['mongodb'])[database][collection]
    collection.single_thread = True
    print('This is the number of documents')
    print(collection.count_documents({}))
    collection.process_documents_with_model(
        model_name=model_name,
        ids=ids,
        batch_size=batch_size,
        verbose=verbose
    )

    import time
    time.sleep(30)
