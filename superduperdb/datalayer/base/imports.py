import importlib
import superduperdb as s
import typing as t


def get_database_from_database_type(database_type: str, database_name: str) -> t.Any:
    """
    Import the database connection from ``superduperdb``

    :param database_type: type of database (supported: ['mongodb'])
    :param database_name: collection of database
    """
    module = importlib.import_module(f'superduperdb.datalayer.{database_type}.client')

    try:
        cfg = getattr(s.CFG, database_type)
    except AttributeError:
        kwargs = {}
    else:
        kwargs = cfg.dict()

    client = module.SuperDuperClient(**kwargs)
    return client.get_database_from_name(database_name)
