from superduper import logging

import time

def my_func(x):
    logging.info(f'my_func({x}) executing')
    time.sleep(1)
    logging.info(f'my_func({x}) executing... DONE')
    return x


def my_other_func(x):
    logging.info(f'my_other_func({x}) executing')
    time.sleep(1)
    logging.info(f'my_other_func({x}) executing... DONE')
    return x


