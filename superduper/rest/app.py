import yaml

from superduper import CFG, logging
from superduper.rest.base import SuperDuperApp
from superduper.rest.build import build_rest_app

assert isinstance(CFG.rest.uri, str), "config.rest.uri should be set with a valid uri"
port = int(CFG.rest.uri.split(':')[-1])

if CFG.rest.config is not None:
    try:
        with open(CFG.rest.config) as f:
            CONFIG = yaml.safe_load(f)
    except FileNotFoundError:
        logging.warn("rest.config should be set with a valid path")
        CONFIG = {}

app = SuperDuperApp('rest', port=port)

build_rest_app(app)
