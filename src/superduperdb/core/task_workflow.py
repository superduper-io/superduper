import dataclasses as dc
import typing as t

import networkx

from .job import Job


@dc.dataclass
class TaskWorkflow:
    database: t.Any
    G: networkx.DiGraph = dc.field(default_factory=networkx.DiGraph)

    def add_edge(self, node1, node2) -> None:
        self.G.add_edge(node1, node2)

    def add_node(self, node, job) -> None:
        self.G.add_node(node, job=job)

    def dependencies(self, node):
        return [self.G.nodes[a]['job'].future for a in self.G.predecessors(node)]

    def __call__(self, db=None, remote: bool = False):
        current_group = [n for n in self.G.nodes if not networkx.ancestors(self.G, n)]
        done = []
        while current_group:
            for node in current_group:
                job: Job = self.G.nodes[node]['job']
                job(db=db, dependencies=self.dependencies(node), remote=remote)
                done.append(node)

            current_group = [
                n
                for n in self.G.nodes
                if set(self.G.predecessors(n)).issubset(set(done))
                and n not in set(done)
            ]
        return self
