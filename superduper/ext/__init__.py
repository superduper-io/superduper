from superduper import logging


def _warn_plugin_deprecated(name):
    message = (
        f'superduper.ext.{name} is deprecated '
        'and will be removed in a future release.'
        f'Please insteall superduper_{name} and use'
        f'from superduper_{name} import * instead.'
    )
    logging.warn(message)
