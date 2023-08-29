from time import sleep as _sleep
from time import time

import superduperdb as s

__all__ = 'sleep', 'time'

TOTAL_SLEEP = 0.0
SAVE_TIME_FILE = True
TIME_FILE = s.ROOT / 'time.txt'


def sleep(t: float) -> None:
    global TOTAL_SLEEP
    TOTAL_SLEEP += t
    _sleep(t)

    if SAVE_TIME_FILE:
        TIME_FILE.write_text(f'{TOTAL_SLEEP}\n')
