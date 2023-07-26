from __future__ import annotations
import dataclasses as dc
import typing as t

import networkx

from superduperdb.misc.configs import CFG

from .job import Job
from distributed.client import Future
from superduperdb.core.job import ComponentJob, FunctionJob

if t.TYPE_CHECKING:
    from superduperdb.datalayer.base.datalayer import Datalayer


@dc.dataclass
class TaskWorkflow:
    database: t.Any
    G: networkx.DiGraph = dc.field(default_factory=networkx.DiGraph)

    def add_edge(self, node1: str, node2: str) -> None:
        self.G.add_edge(node1, node2)

    def add_node(self, node: str, job: t.Union[FunctionJob, ComponentJob]) -> None:
        self.G.add_node(node, job=job)

    def dependencies(self, node: str) -> t.List[t.Optional[t.Union[t.Any, Future]]]:
        return [self.G.nodes[a]['job'].future for a in self.G.predecessors(node)]

    def watch(self):
        for node in list(networkx.topological_sort(self.G)):
            self.G.nodes[node]['job'].watch()

    def __call__(
        self, db: t.Optional[Datalayer] = None, distributed: bool = False
    ) -> 'TaskWorkflow':
        if distributed is None:
            distributed = CFG.distributed

        current_group = [n for n in self.G.nodes if not networkx.ancestors(self.G, n)]
        done = []
        while current_group:
            for node in current_group:
                job: Job = self.G.nodes[node]['job']
                job(
                    db=db, dependencies=self.dependencies(node), distributed=distributed
                )
                done.append(node)
            current_group = [
                n
                for n in self.G.nodes
                if set(self.G.predecessors(n)).issubset(set(done))
                and n not in set(done)
            ]
        return self
