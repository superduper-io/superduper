import datetime
import uuid
from collections import defaultdict

import networkx
from celery import chain, group


class TaskWorkflow:
    def __init__(self, database):
        self.G = networkx.DiGraph()
        self._path_lengths = None
        self.database = database

    def add_edge(self, node1, node2):
        self.G.add_edge(node1, node2)

    def add_node(self, node, data=None):
        data = data or {}
        self.G.add_node(node)
        for k, v in data.items():
            self.G[node][k] = v

    def __add__(self, other):
        for node in other.nodes:
            self.G.add_node(node)
        roots_other = []
        for node in other:
            if not networkx.ancestors(node):
                roots_other.append(node)
        leafs_this = []
        for node in self.G:
            if not networkx.descendants(node):
                leafs_this.append(node)
        for node1 in leafs_this:
            for node2 in roots_other:
                self.add_edge(node1, node2)
        return self

    @property
    def path_lengths(self):
        if self._path_lengths is None:
            self._path_lengths = {}
            for node in networkx.topological_sort(self.G):
                if not self._path_lengths:
                    self._path_lengths[node] = 0
                else:
                    self._path_lengths[node] = min(self._path_lengths[n]
                                                   for n in self.G.predecessors(node) + 1)
        return self._path_lengths

    def compile(self):
        from superduperdb.getters.jobs import function_job
        to_chain = []
        lookup = defaultdict(lambda: [])
        for n in self.G.nodes:
            lookup[self.path_lengths[n]].append(n)
        max_path_lengths = max(lookup.keys())
        for i in range(max_path_lengths):
            current_nodes = lookup[i]
            job_id = str(uuid.uuid4())
            current_group = []
            for node in current_nodes:
                self.database._create_job_record({
                    'identifier': job_id,
                    'time': datetime.datetime.now(),
                    'status': 'pending',
                    'method': node.data['task'].__name__,
                    'args': node.data['args'],
                    'kwargs': node.data['kwargs'],
                    'stdout': [],
                    'stderr': [],
                })
                self.G[node]['job_id'] = job_id
                current_group.append(
                    function_job.signature(
                        args=[
                            self.database._database_type,
                            self.database.name,
                            node.data['task'].__name__,
                            node.data['args'],
                            node.data['kwargs'],
                            job_id,
                        ],
                        task_id=job_id,
                    )
                    for node in current_nodes
                )
            current_group = group(*current_group)
            to_chain.append(current_group)
        entire_workflow = chain(*to_chain)
        return entire_workflow

    def call_node(self, n):
        n['task'](*n['args'], **n['kwargs'])

    def __call__(self, *args, asynchronous=False, **kwargs):
        if not asynchronous:
            nodes = [n for n in self.G.nodes if not n.ancestors]
            while nodes:
                new_nodes = []
                for n in nodes:
                    self.call_node(n)
                    new_nodes.extend([nn for nn in n.children])
                nodes = new_nodes
        else:
            workflow = self.compile()
            workflow.apply_async()
