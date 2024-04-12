MONGOMOCK_URI = 'mongomock:///test_db'
SQL_URI = 'sqlite://'
# We can use the following SQL_URI for testing with a different database:
# SQL_URI = 'clickhouse://default:@localhost:8123/default'
# SQL_URI = 'mysql://root:root123@localhost:3306/test_db'
N_DATA_POINTS = 5


class DBConfig:
    # Common configuration parameters
    COMMON_CONFIG = {
        'empty': False,
        'add_encoders': True,
        'add_data': True,
        'add_models': True,
        'add_vector_index': True,
        'n_data': N_DATA_POINTS,
        'add_query': True
    }

    # Base configuration for MongoDB and SQL databases
    _mongodb_base = {
        'db_type': 'mongodb',
        'data_backend': MONGOMOCK_URI,
        **COMMON_CONFIG,
    }
    _sqldb_base = {'db_type': 'sqldb', 'data_backend': SQL_URI, **COMMON_CONFIG}

    # Configurations for an empty database
    mongodb_empty = {**_mongodb_base, 'empty': True}
    sqldb_empty = {**_sqldb_base, 'empty': True}

    # Full database configurations including encoder, data, model, and vector_index
    mongodb = {**_mongodb_base}
    sqldb = {**_sqldb_base}

    # Configurations without vector_index
    mongodb_no_vector_index = {**_mongodb_base, 'add_vector_index': False}
    sqldb_no_vector_index = {**_sqldb_base, 'add_vector_index': False}

    # Configurations with only encoder and data
    mongodb_data = {**_mongodb_base, 'add_models': False, 'add_vector_index': False}
    sqldb_data = {**_sqldb_base, 'add_models': False, 'add_vector_index': False}

    # Additional frequently used presets can be added as needed...
