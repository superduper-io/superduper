import importlib

from superduperdb import CFG


def get_database_from_database_type(database_type, database_name):
    """
    Import the database connection from ``superduperdb``

    :param database_type: type of database (supported: ['mongodb'])
    :param database_name: name of database
    """
    module = importlib.import_module(f'superduperdb.datalayer.{database_type}.client')
    client_cls = getattr(module, 'SuperDuperClient')

    try:
        cfg = getattr(CFG, database_type)
    except AttributeError:
        kwargs = {}
    else:
        kwargs = cfg.dict()

    client = client_cls(**kwargs)
    return client.get_database_from_name(database_name)
