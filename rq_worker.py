from redis import Redis
from rq import Worker

# Preload libraries
import os
import sys

sys.path.insert(0, os.getcwd())

# Provide the worker with the list of queues (str) to listen to.
w = Worker(['default'], connection=Redis())
w.work()