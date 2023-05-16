import os

from superduperdb import cf
from superduperdb.misc.logger import logging


def init_fn():
    logging.info('Setting OpenAI api-key...')
    os.environ['OPENAI_API_KEY'] = cf['apis']['providers']['openai']['api_key']
