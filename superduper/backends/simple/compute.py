import contextlib
import sys
import threading
import time
import typing as t
from collections import defaultdict
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from queue import Queue
from threading import Lock, Thread

import requests

from superduper import logging
from superduper.backends.base.compute import ComputeBackend
from superduper.backends.simple.vector_search import SimpleVectorSearchClient
from superduper.base.metadata import Job

if t.TYPE_CHECKING:
    from superduper import Component
    from superduper.base.datalayer import Datalayer


class ThreadStdoutRouter:
    """A pseudo-file that forwards writes to a thread-specific file object."""

    def __init__(self):
        self._local = threading.local()  # holds .stream per thread

    # –– API that print() expects ––
    def write(self, data):
        stream = getattr(self._local, "stream", sys.__stdout__)
        stream.write(data)

    def flush(self):
        stream = getattr(self._local, "stream", sys.__stdout__)
        stream.flush()

    # –– helper for threads ––
    @contextlib.contextmanager
    def redirect_to(self, path, mode="w"):
        f = open(path, mode)
        self._local.stream = f
        try:
            yield
        finally:
            f.close()
            del self._local.stream


ROUTER = ThreadStdoutRouter()


class SimpleComputeClient(ComputeBackend):
    """Simple compute client for local execution."""

    def __init__(self):
        super().__init__()
        self.uri = 'http://localhost:8001'

    def release_futures(self, context: str):
        """Release futures from backend.

        :param context: Futures context to release.
        """
        requests.post(
            f'{self.uri}/release_futures?context={context}',
        )

    def disconnect(self) -> None:
        """Disconnect the client."""
        pass

    def remote(self, uuid: str, function: str, *args, **kwargs):
        """
        Call a remote function on the server.

        :param uuid: The UUID of the component.
        :param function: The function to call.
        :param args: The arguments to pass to the function.
        :param kwargs: The keyword arguments to pass to the function.
        """
        raise NotImplementedError

    def initialize(self):
        """Connect to address."""

    def put_component(self, component: str, uuid: str):
        """Create handler on component declare.

        :param component: Component to put.
        """

    @property
    def db(self) -> 'Datalayer':
        """Get the ``db``."""
        return self._db

    @db.setter
    def db(self, value: 'Datalayer'):
        """Set the ``db``.

        :param value: ``Datalayer`` instance.
        """
        self._db = value

    def drop_component(
        self, component: str, identifier: str | None = None, uuid: str | None = None
    ):
        """Drop the component from compute.

        :param component: Component name.
        :param identifier: Component identifier.
        """

    def submit(self, job: Job) -> str:
        """Submits a function to simple compute backend."""
        r = requests.post(f'{self.uri}/submit', json=job.dict())
        if r.status_code != 200:
            raise Exception(f'Failed to submit job: {r.text}')
        return job.job_id

    def drop(self):
        """Drop the compute backend."""
        r = requests.post(
            f'{self.uri}/drop',
        )
        if r.status_code != 200:
            raise Exception(f'Failed to submit job: {r.text}')

    def list_components(self):
        return []

    def list_uuids(self):
        return []


class StopEvent:
    """A simple event to stop the worker thread."""


def job_worker(job: Job) -> t.Any:
    """Worker function to execute a job."""
    from superduper import superduper

    logging.info('Building job datalayer instance')

    db = superduper(cluster_engine='local')
    db.cluster.vector_search = SimpleVectorSearchClient()

    logging.info('Building job datalayer instance... DONE')

    logging.info('Executing job in worker')
    import os

    # current = os.environ.copy()
    # os.environ.update(job.envs)
    # for p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    #     if p and p not in sys.path:
    #         sys.path.insert(0, p)
    logging.info('Redirecting stdout and stderr')
    os.makedirs(os.path.expanduser('~/.superduper/simple'), exist_ok=True)
    with ROUTER.redirect_to(
        os.path.expanduser(f'~/.superduper/simple/{job.huuid}.log'), 'w'
    ):
        result = job.run(db)
    # os.environ.clear()
    # os.environ.update(current)
    logging.info('Executing job in worker... DONE')
    db.disconnect()
    return result


class SimpleComputeBackend(ComputeBackend):
    """
    A mockup backend for running jobs locally.

    :param uri: Optional uri param.
    :param kwargs: Optional kwargs.
    """

    def __init__(
        self,
        uri: t.Optional[str] = None,
        **kwargs,
    ):
        self.uri = uri
        self.kwargs = kwargs
        self._cache: t.Dict = {}
        self._db = None
        self.futures: t.DefaultDict = defaultdict(lambda: {})
        self.contexts: t.Dict = {}
        self.locks: t.Dict = {}

    def release_futures(self, context: str):
        """Release futures for a given context.

        :param context: The apply context to release futures for.
        """
        self.contexts[context].put(StopEvent())

    def submit(self, job: Job) -> str:
        """
        Submits a function for local execution.

        :param job: The `Job` to be executed.
        """
        if job.context not in self.contexts:
            self.contexts[job.context] = Queue()
            self.locks[job.context] = Lock()
            self._start_context_thread(job.context)

        logging.info(f'Adding job to context queue {job.context}')
        self.contexts[job.context].put(job)
        logging.info(f'Adding job to context queue {job.context}... DONE')
        return job.job_id

    def _execute_job(self, job: Job, context: str, executor: ThreadPoolExecutor):
        """Execute a job."""
        upstream = [self.futures[context][dep] for dep in job.dependencies]
        done, _ = wait(upstream, return_when=ALL_COMPLETED)
        errors = [f.exception() for f in done if f.exception() is not None]
        if errors:
            logging.error('\n'.join([str(e) for e in errors]))
            raise Exception(
                f'Upstream job dependencies failed with {len(errors)} errors'
            )

        logging.info(f'Submitting job {job.huuid} in context {context}')
        fut = executor.submit(job_worker, job)
        self.futures[context][job.job_id] = fut

        def _log_errors(f):
            exc = f.exception()
            if exc is not None:
                logging.exception(f"Job {job.huuid} failed in {context}", e=exc)

        fut.add_done_callback(_log_errors)

        logging.info(f'Submitting job {job.huuid} in context {context}... DONE')

    def _wait_for_jobs_to_end(self, context: str):
        futures = list(self.futures[context].values())
        done, _ = wait(futures, return_when=ALL_COMPLETED)

        # Collect any exceptions
        errors = [f.exception() for f in done if f.exception() is not None]
        if errors:
            logging.error('\n'.join([str(e) for e in errors]))
            raise Exception(f'Job failed with {len(errors)} errors')

    def drop(self):
        """Drop the compute."""
        for context in list(self.contexts.keys()):
            self.release_futures(context)

        timeout = 5
        while self.contexts:
            time.sleep(0.1)
            if time.time() - timeout > 0:
                raise TimeoutError('Timeout waiting for jobs to finish')

    def _start_context_thread(self, context: str):
        """Start a thread for the given context."""

        def worker():
            with ThreadPoolExecutor() as executor:
                while True:
                    logging.info(f'Waiting for event in context {context}')
                    event: Job | StopEvent = self.contexts[context].get()
                    logging.info(f'Received event in context {context}; {event}')
                    if isinstance(event, StopEvent):
                        logging.info('Received stop event, stopping worker thread')
                        logging.info('Waiting for jobs to finish')
                        self._wait_for_jobs_to_end(context)
                        logging.info('Waiting for jobs to finish... DONE')
                        logging.info('Cleaning up contexts, futures and locks')
                        with self.lock:
                            del self.contexts[context]
                            del self.futures[context]
                            del self.locks[context]
                        logging.info('Cleaning up contexts, futures and locks... DONE')
                        break
                    else:
                        logging.info(f'Received job {event.huuid}, ...executing')
                        self._execute_job(event, context, executor)
                        logging.info(
                            f'Received job {event.huuid}, ...executing... DONE'
                        )
            logging.info('Received stop event, stopping worker thread... DONE')

        thread = Thread(target=worker)
        thread.start()

    def list_components(self):
        """List all components on the compute."""
        return []

    def list_uuids(self):
        """List all UUIDs on the compute."""
        return []

    def drop_component(
        self, component: str, identifier: str | None = None, uuid: str | None = None
    ):
        """Drop a component from the compute."""

    def put_component(self, component: str, uuid: str):
        """Create a handler on compute."""

    def initialize(self):
        """Initialize the compute."""

    def remote(self, uuid: str, component: str, method: str, *args, **kwargs):
        """Run a remote method on the compute."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """Disconnect the local client."""

    def build(self, app):
        import inspect

        @app.post('/submit')
        def submit(job: t.Dict):
            logging.info(f'Processing submit request {job}')
            parameters = inspect.signature(Job).parameters
            job = {k: v for k, v in job.items() if k in parameters}
            self.submit(Job(**job))
            logging.info(f'Processing submit request {job}... DONE')
            return {"status": "success"}

        @app.post('/release_futures')
        def release_futures(context: str):
            self.release_futures(context)
            return {"status": "success"}
