from sddb.client import SddbClient


def process_documents_with_model(
    database,
    collection,
    model_name,
    ids,
    batch_size=10,
    verbose=False
):
    collection = SddbClient()[database][collection]
    collection.process_documents_with_model(
        model_name=model_name,
        ids=ids,
        batch_size=batch_size,
        verbose=verbose
    )
