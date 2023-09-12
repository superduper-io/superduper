import queue
import threading
import traceback
import typing as t

from superduperdb import logging
from superduperdb.container.job import FunctionJob
from superduperdb.container.serializable import Serializable
from superduperdb.container.task_workflow import TaskWorkflow
from superduperdb.db.base.db import DB

from .base import DBEvent, Packet
from .task_queue import cdc_queue
from .vector_task_factory import vector_task_factory


class CDCHandler(threading.Thread):
    """
    This class is responsible for handling the change by executing the taskflow.
    This class also extends the task graph by adding funcation job node which
    does post model executiong jobs, i.e `copy_vectors`.
    """

    _QUEUE_BATCH_SIZE: int = 100
    _QUEUE_TIMEOUT: float = 0.01

    def __init__(self, db: DB, stop_event: threading.Event):
        """__init__.

        :param db: a superduperdb instance.
        :param stop_event: A threading event flag to notify for stoppage.
        """
        self.db = db
        self._stop_event = stop_event

        threading.Thread.__init__(self, daemon=False)

    def submit_task_workflow(
        self, cdc_query: t.Optional[Serializable], ids: t.Sequence, task: str = "copy"
    ) -> None:
        """submit_task_workflow.
        A fxn to build a taskflow and execute it with changed ids.
        This also extends the task workflow graph with a node.
        This node is responsible for applying a vector indexing listener,
        and copying the vectors into a vector search database.

        :param cdc_query: A query which will be used by `db._build_task_workflow` method
        to extract the desired data.
        :param ids: List of ids which were observed as changed document.
        :param task: A task name to be executed on vector db.
        """
        if task == "delete":
            task_workflow = TaskWorkflow(self.db)
        else:
            task_workflow = self.db._build_task_workflow(
                cdc_query, ids=ids, verbose=False
            )

        task_workflow = self.create_vector_listener_task(
            task_workflow, cdc_query=cdc_query, ids=ids, task=task
        )
        task_workflow.run_jobs()

    def create_vector_listener_task(
        self,
        task_workflow: TaskWorkflow,
        cdc_query: t.Optional[Serializable],
        ids: t.Sequence[str],
        task: str = 'copy',
    ) -> TaskWorkflow:
        """create_vector_listener_task.
        A helper function to define a node in taskflow graph which is responsible for
        executing the defined ``task`` on a vector db.

        :param task_workflow: A DiGraph task flow which defines task on a di graph.
        :param db: A superduperdb instance.
        :param cdc_query: A basic find query to get cursor on collection.
        :param ids: A list of ids observed during the change
        :param task: A task name to be executed on vector db.
        """
        from superduperdb.container.vector_index import VectorIndex

        task_callable, task_name = vector_task_factory(task=task)
        serialized_cdc_query = cdc_query.serialize() if cdc_query else None
        for identifier in self.db.show('vector_index'):
            vector_index = self.db.load(identifier=identifier, type_id='vector_index')
            vector_index = t.cast(VectorIndex, vector_index)
            assert not isinstance(vector_index.indexing_listener, str)
            indexing_listener_identifier = vector_index.indexing_listener.identifier
            task_workflow.add_node(
                f'{task_name}({indexing_listener_identifier})',
                job=FunctionJob(
                    callable=task_callable,
                    args=[indexing_listener_identifier, serialized_cdc_query, ids],
                    kwargs={},
                ),
            )
        if task != 'delete':
            assert indexing_listener_identifier
            model, _, key = indexing_listener_identifier.rpartition('/')
            task_workflow.add_edge(
                f'{model}.predict({key})',
                f'{task_name}({indexing_listener_identifier})',
            )
        return task_workflow

    def on_create(self, packet: Packet) -> None:
        ids = packet.ids
        cdc_query = packet.query
        self.submit_task_workflow(cdc_query=cdc_query, ids=ids)

    def on_update(self, packet: Packet) -> None:
        self.on_create(packet)

    def on_delete(self, packet: Packet) -> None:
        ids = packet.ids
        cdc_query = packet.query
        self.submit_task_workflow(cdc_query=cdc_query, ids=ids, task="delete")

    def _handle(self, packet: Packet) -> None:
        if packet.event_type == DBEvent.insert:
            self.on_create(packet)
        elif packet.event_type == DBEvent.update:
            self.on_update(packet)
        elif packet.event_type == DBEvent.delete:
            self.on_delete(packet)

    def get_batch_from_queue(self):
        """
        Get a batch of packets from task queue, with a timeout.
        """
        packets = []
        try:
            for _ in range(self._QUEUE_BATCH_SIZE):
                packets.append(cdc_queue.get(block=True, timeout=self._QUEUE_TIMEOUT))
                if self._stop_event.is_set():
                    return 0

        except queue.Empty:
            if len(packets) == 0:
                return None
        return Packet.collate(packets)

    def run(self):
        while not self._stop_event.is_set():
            try:
                packets = self.get_batch_from_queue()
                if packets:
                    self._handle(packets)
                if packets == 0:
                    break
            except Exception as exc:
                traceback.print_exc()
                logging.info(f'Error while handling cdc batches :: reason {exc}')
