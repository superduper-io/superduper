import warnings

from .configs import CFG

__all__ = ('logging',)


if CFG.logging.type == 'stdout':

    class logging:
        warn = staticmethod(warnings.warn)
        error = staticmethod(warnings.warn)
        debug = staticmethod(print)
        info = staticmethod(print)

    def dont_print(*a, **ka):
        pass

    level = CFG.logging.level

    if level == level.DEBUG:
        pass

    elif level == level.INFO:
        logging.debug = dont_print  # type: ignore[assignment]

    else:
        logging.debug = dont_print  # type: ignore[assignment]
        logging.info = dont_print  # type: ignore[assignment]

else:
    import logging  # type: ignore[assignment]

    level = getattr(logging, CFG.logging.level.name)
    logging.basicConfig(level=level, **CFG.logging.kwargs)  # type: ignore[attr-defined]
    logging.getLogger('distributed').propagate = True  # type: ignore[attr-defined]
    logging.getLogger('vcr').setLevel(logging.WARNING)  # type: ignore[attr-defined]
