import logging
from superduperdb import cf
from warnings import warn

cf_logging = cf.get('logging', {'type': 'stdout', 'level': 'INFO'})

if cf_logging['type'] == 'stdout':
    logging = lambda: None
    logging.info = (
        print if cf_logging['level'] in {'DEBUG', 'INFO'} else lambda x: None
    )
    logging.debug = print if cf_logging['level'] == 'DEBUG' else lambda x: None
    logging.warn = (
        warn
        if cf_logging['level'] in {'DEBUG', 'INFO', 'WARN'}
        else lambda x: None
    )
    logging.error = warn
else:
    cf_logging = cf.get('logging', {})
    logging.basicConfig(
        level=getattr(logging, cf_logging.get('level', 'INFO')),
        **{k: v for k, v in cf_logging.items() if k != 'level'},
    )
    logging.getLogger("distributed").propagate = True
