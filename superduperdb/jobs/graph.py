import datetime
import uuid
from collections import defaultdict

import networkx
from celery import chain, group

from superduperdb.serving.utils import encode_ids_parameters


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
            self.G.nodes[node][k] = v

    def __mul__(self, other):
        for node in other.nodes:
            self.G.add_node(node)
        for edge in other.edges:
            self.G.add_edge(*edge)
        return self

    def __add__(self, other):
        for node in other.nodes:
            self.G.add_node(node)
        for edge in other.edges:
            self.G.add_edge(*edge)
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
                    self._path_lengths[node] = min([self._path_lengths[n]
                                                    for n in self.G.predecessors(node)]) + 1
        return self._path_lengths

    def compile(self, dry_run=False):
        from superduperdb.serving.jobs import function_job
        to_chain = []
        lookup = defaultdict(lambda: [])
        for n in self.G.nodes:
            lookup[self.path_lengths[n]].append(n)
        max_path_lengths = max(lookup.keys())
        for i in range(max_path_lengths + 1):
            current_nodes = lookup[i]
            job_id = str(uuid.uuid4())
            current_group = []
            for node in current_nodes:
                node_object = self.G.nodes[node]
                if not dry_run:
                    self.database._create_job_record({
                        'identifier': job_id,
                        'time': datetime.datetime.now(),
                        'status': 'pending',
                        'method': node_object['task'].__name__,
                        'args': node_object['args'],
                        'kwargs': node_object['kwargs'],
                        'stdout': [],
                        'stderr': [],
                    })
                self.G.nodes[node]['job_id'] = job_id
                args, kwargs = self.get_args_kwargs(node_object)
                current_group.append(
                    function_job.signature(
                        args=[
                            self.database._database_type,
                            self.database.name,
                            node_object['task'].__name__,
                            args,
                            {**kwargs, 'remote': False},
                            job_id,
                        ],
                        task_id=job_id,
                    )
                )
            current_group = group(*current_group)
            to_chain.append(current_group)
        entire_workflow = chain(*to_chain)
        return entire_workflow

    def get_args_kwargs(self, n):
        return encode_ids_parameters(
            n['args'], n['kwargs'], n['task'].positional_convertible,
            n['task'].keyword_convertible,
        )

    def call_node(self, n):
        return n['task'](*n['args'], **n['kwargs'])

    def __call__(self, *args, remote=False, **kwargs):
        if not remote:
            nodes = [n for n in self.G.nodes if not networkx.ancestors(self.G, n)]
            while nodes:
                new_nodes = []
                for n in nodes:
                    self.call_node(self.G.nodes[n])
                    new_nodes.extend([nn for nn in networkx.descendants(self.G, n)])
                nodes = new_nodes
        else:
            workflow = self.compile()
            workflow.apply_async()
