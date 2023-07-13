from warnings import warn

from superduperdb import CFG

__all__ = ('logging',)


if CFG.logging.type == 'stdout':  # type: ignore[has-type]

    def logging():
        pass

    def dont_print(*a, **ka):
        pass

    logging.error = logging.warn = warn
    level = CFG.logging.level  # type: ignore[has-type]

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

    level = getattr(logging, CFG.logging.level.name)  # type: ignore[has-type]
    logging.basicConfig(level=level, **CFG.logging.kwargs)  # type: ignore[has-type]
    logging.getLogger('distributed').propagate = True
