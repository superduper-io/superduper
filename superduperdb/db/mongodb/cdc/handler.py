import queue
import threading
import traceback

from superduperdb import logging
from superduperdb.container.job import FunctionJob
from superduperdb.container.task_workflow import TaskWorkflow
from superduperdb.db.base.db import DB
from superduperdb.misc.runnable.queue_chunker import QueueChunker
from superduperdb.misc.runnable.runnable import Event

from .base import Packet
from .vector_task_factory import vector_task_factory

queue_chunker = QueueChunker(chunk_size=100, timeout=0.2)
CDC_QUEUE: queue.Queue = queue.Queue()


class CDCHandler(threading.Thread):
    """
    This class is responsible for handling the change by executing the taskflow.
    This class also extends the task graph by adding funcation job node which
    does post model executiong jobs, i.e `copy_vectors`.
    """

    def __init__(self, db: DB, stop_event: Event):
        """__init__.

        :param db: a superduperdb instance.
        :param stop_event: A threading event flag to notify for stoppage.
        """
        self.db = db
        self._stop_event = stop_event

        threading.Thread.__init__(self, daemon=False)

    def run(self):
        try:
            for c in queue_chunker(CDC_QUEUE, self._stop_event):
                _submit_task_workflow(self.db, Packet.collate(c))

        except Exception as exc:
            traceback.print_exc()
            logging.info(f'Error while handling cdc batches :: reason {exc}')


def _submit_task_workflow(db: DB, packet: Packet) -> None:
    """
    Build a taskflow and execute it with changed ids.

    This also extends the task workflow graph with a node.
    This node is responsible for applying a vector indexing listener,
    and copying the vectors into a vector search database.
    """
    if packet.is_delete:
        workflow = TaskWorkflow(db)
    else:
        workflow = db._build_task_workflow(packet.query, ids=packet.ids, verbose=False)

    task = 'delete' if packet.is_delete else 'copy'
    task_callable, task_name = vector_task_factory(task=task)

    serialized_query = packet.query.serialize() if packet.query else None

    def add_node(identifier):
        from superduperdb.container.vector_index import VectorIndex

        vi = db.load(identifier=identifier, type_id='vector_index')
        assert isinstance(vi, VectorIndex)

        assert not isinstance(vi.indexing_listener, str)
        listener_id = vi.indexing_listener.identifier

        args = [listener_id, serialized_query, packet.ids]
        job = FunctionJob(callable=task_callable, args=args)
        workflow.add_node(f'{task_name}({listener_id})', job=job)

        return listener_id

    listener_ids = [add_node(i) for i in db.show('vector_index')]
    if not packet.is_delete:
        assert listener_ids
        listener_id = listener_ids[-1]
        model, _, key = listener_id.rpartition('/')
        workflow.add_edge(f'{model}.predict({key})', f'{task_name}({listener_id})')

    workflow.run_jobs()
