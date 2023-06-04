from superduperdb import CFG
from warnings import warn

__all__ = 'logging',

if CFG.logging.type == 'stdout':
    def logging():
        pass

    def dont_print(*a, **ka):
        pass

    logging.error = logging.warn = warn
    level = CFG.logging.level

    if level == level.DEBUG:
        logging.debug = print
        logging.info = print

    elif level == level.INFO:
        logging.debug = dont_print
        logging.info = print

    else:
        logging.debug = dont_print
        logging.info = dont_print

else:
    import logging

    level = getattr(logging, CFG.logging.level.name)
    logging.basicConfig(level=level, **CFG.logging.kwargs)
    logging.getLogger('distributed').propagate = True
