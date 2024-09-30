import yaml
from superduper import CFG, logging

from superduper.rest.build import build_rest_app
from superduper.rest.base import app as superduperapp

assert isinstance(
    CFG.cluster.rest.uri, str
), "cluster.rest.uri should be set with a valid uri"
port = int(CFG.cluster.rest.uri.split(':')[-1])

if CFG.cluster.rest.config is not None:
    try:
        with open(CFG.cluster.rest.config) as f:
            CONFIG = yaml.safe_load(f)
    except FileNotFoundError:
        logging.warn("cluster.rest.config should be set with a valid path")
        CONFIG = {}

app = superduperapp.SuperDuperApp('rest', port=port)

build_rest_app(app)
