import os

from superduperdb.apis import api_cf
from superduperdb.misc.logger import logging


def init_fn():
    logging.info('Setting OpenAI api-key...')
    os.environ['OPENAI_API_KEY'] = api_cf['providers']['openai']['api_key']
