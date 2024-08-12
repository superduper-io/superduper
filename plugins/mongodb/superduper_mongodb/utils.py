from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from superduper import logging
from superduper.misc.anonymize import anonymize_url


def get_avaliable_conn(uri: str, **kwargs):
    """Get an available connection to the database.

    This can avoid some issues with database permission verification.
    1. Try to connect to the database with the given URI.
    2. Try to connect to the database with the base URI without database name.

    :param uri: The URI of the database.
    :param kwargs: Additional keyword arguments for the MongoClient.
    """
    base_uri, db_name = uri.rsplit("/", 1)
    kwargs.setdefault("serverSelectionTimeoutMS", 5000)

    raise_e = None

    # Try to connect to the database with the given URI
    client: MongoClient = MongoClient(uri, **kwargs)
    try:
        client[db_name].list_collection_names()
        return client
    except ServerSelectionTimeoutError as e:
        # If the server is not available, raise the exception
        raise e
    except Exception as e:
        uri_mask = anonymize_url(uri)
        logging.warn(
            f"Failed to connect to the database with the given URI: {uri_mask}"
        )
        logging.error(str(e))
        raise_e = e
        client.close()

    # Try to connect to the database with base URI without database name
    client = MongoClient(base_uri, **kwargs)
    base_uri_mask = anonymize_url(base_uri)
    try:
        logging.info(
            f"Trying to connect to the database with the base URI: {base_uri_mask}"
        )
        client[db_name].list_collection_names()
        return client
    except Exception as e:
        logging.warn(
            f"Failed to connect to the database with the base URI: {base_uri_mask}"
        )
        logging.error(str(e))
        client.close()

    if raise_e:
        logging.error("Failed to connect to the database")
        raise raise_e
