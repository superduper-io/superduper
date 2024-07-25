import os

basedir = os.path.abspath(os.path.dirname(__file__))

SUPERDUPER_CONFIG = os.environ.get("SUPERDUPER_CONFIG", "test/configs/default.yaml")

os.environ["SUPERDUPER_CONFIG"] = os.path.join(basedir, "config.yaml")
