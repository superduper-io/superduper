import dataclasses as dc
import typing as t
from collections import OrderedDict
from functools import cached_property
from inspect import signature

import networkx as nx

from superduperdb.base.artifact import Artifact
from superduperdb.components.component import Component
from superduperdb.components.model import Model
from superduperdb.ext.torch.model import TorchModel

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


class KeyStore:
    def __init__(self, model, params):
        self.model = model
        self._params = params
        for param in params:
            setattr(self, param, f'{model}.{param}')

    def __len__(self):
        return len(self._params)

    @property
    def params(self):
        return self._params


class RootModel:
    def __init__(self):
        self.input = KeyStore('root', [])

    def predict(self, x):
        return x


class GraphModel:
    def __init__(self, model: t.Union['Model', 'TorchModel']):
        self.identifier = model.identifier
        self.output = f'{model.identifier}.output'

        predict_method = model.predict_method
        if isinstance(model, TorchModel):
            predict_method = model.forward_method

        assert isinstance(model.object, Artifact)
        self.object = model.object.artifact

        if predict_method:
            self.predict_method = getattr(self.object, predict_method)
        else:
            self.predict_method = self.object

        params = self.set_input()
        self.input = KeyStore(model.identifier, params)

    def set_input(self):
        sig = signature(self.predict_method)
        return list(sig.parameters.keys())

    def predict(self, *args, **kwargs):
        return self.predict_method(*args, **kwargs)


class Node:
    def __init__(self, name, db=None):
        self.name = name

        if name == Graph._input:
            self.model = RootModel()
        else:
            self.model = GraphModel(db.load('model', name))
        self._output = None

    @cached_property
    def params(self):
        return self.model.input.params

    @cached_property
    def required_input(self):
        return len(self.params)

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, output):
        if self._output:
            raise ValueError('Not allowed to set output of a graph node')
        self._output = output

    def predict(self, *args, **kwargs):
        self.output = self.model.predict(*args, **kwargs)
        return self.output


@dc.dataclass(kw_only=True)
class Graph(Component):
    db: 'Datalayer'
    _DEFAULT_ARG_WEIGHT: t.ClassVar[str] = '_base'
    _input: t.ClassVar[str] = '_Graph_input'

    def __post_init__(self):
        self.G = nx.DiGraph()
        self._key_store = {}
        self.nodes = {}
        self._node_output_cache = {}

    def connect(
        self,
        u: t.Union[str, Component],
        v: t.Union[str, Model],
        on: t.Optional[str] = None,
    ):
        assert isinstance(u, (Model, Graph, str))
        assert isinstance(v, (Model, str))

        if isinstance(v, Model):
            v = v.identifier
        if isinstance(u, Model):
            u = u.identifier

        if isinstance(u, Graph):
            u = self._input

        if u not in self.nodes:
            node = Node(u, self.db)
            u = node.name

            self.nodes[u] = node
            self.G.add_node(u)

        if v not in self.nodes:
            node = Node(v, self.db)
            self.nodes[node.name] = node
            self.G.add_node(node.name)
            v = node.name

        G_ = self.G.copy()
        G_.add_edge(u, v, weight=on or self._DEFAULT_ARG_WEIGHT)

        if not nx.is_directed_acyclic_graph(G_):
            raise TypeError('The graph is not DAG with this edge')
        self.G = G_
        return self.nodes[v]

    def stash_node_output(self, node, output):
        self._node_output_cache[node.name] = output

    def group_nodes_by_degree(self, nodes):
        grouped_dict = OrderedDict()
        for item in nodes:
            key = item[1]
            value = item[0]
            if key not in grouped_dict:
                grouped_dict[key] = []
            grouped_dict[key].append(value)
        return grouped_dict

    def level_traversal(self, G, nodes, traversal_path=[]):
        if len(nodes) == 1:
            traversal_path.append(nodes[0])
            return traversal_path
        G = G.subgraph(nodes)

        # Traverse
        nodes = sorted(G.in_degree, key=lambda x: x[1], reverse=False)
        grouped_nodes = self.group_nodes_by_degree(nodes)

        # Find zero in bound degree nodes and add to tranversal_path
        zero_bound_nodes = grouped_nodes.pop(0)
        _ = [traversal_path.append(n) for n in zero_bound_nodes]

        for _, nodes in grouped_nodes.items():
            self.level_traversal(G, nodes, traversal_path)
        return traversal_path

    def traversal(self, graph, nodes, traversal_path=[]):
        if not nodes:
            return traversal_path

        def find_level_neighbors(graph, nodes):
            neighbors = []

            for node in nodes:
                neighbor = list(graph.neighbors(node))
                if neighbor:
                    _ = [neighbors.append(n) for n in neighbor]
            if neighbors:
                neighbors = set(neighbors) - set(nodes)
                neighbors = list(neighbors)
            return neighbors

        neighbors = find_level_neighbors(graph, nodes)

        if not neighbors:
            traversal_path += nodes
            return traversal_path

        S = graph.subgraph(neighbors)
        traversal_path = self.level_traversal(S, neighbors, traversal_path)

        neighbors = find_level_neighbors(graph, neighbors)
        return self.traversal(graph, neighbors, traversal_path)

    def validate(self, path):
        for node in path:
            nodes = list(self.G.predecessors(node))

            arg_nodes = list(map(lambda x: self.nodes[x], nodes))
            node = self.nodes[node]

            if node.required_input != len(arg_nodes):
                raise TypeError(
                    f'Graph disconnected at Node: {node.name} '
                    f'and is partially connected with {nodes}\n'
                    f'Required connected node is {node.required_input} '
                    f'but got only {len(nodes)}, '
                    f'Node required params: {node.params}'
                )

    def predict(self, x, one=True, select=None):
        # TODO: Implement select logic
        if self._input not in self.G.nodes:
            raise TypeError(
                'Root graph node is not present'
                ', make sure to add graph node'
                'with atleast one other node'
            )
        path = self.traversal(self.G, [self._input], [self._input])
        self.validate(path)
        output = None

        for graph_node in path:
            node = self.nodes[graph_node]

            if graph_node == '_Graph_input':
                node.output = x
                continue

            predecessors = list(self.G.predecessors(graph_node))

            arg_nodes = list(map(lambda x: self.nodes[x], predecessors))
            kwargs = {}

            for predecessor, arg in zip(predecessors, arg_nodes):
                data = self.G.get_edge_data(predecessor, graph_node)
                key = data['weight']
                kwargs[key] = arg.output
            args = []
            if self._DEFAULT_ARG_WEIGHT in kwargs:
                args = list(kwargs.values())
                kwargs = {}
            output = node.predict(*args, **kwargs)

        return output

    def show(self):
        path = self.traversal(self.G, [self._input], [self._input])
        path = ' --> '.join(path)
        print(path)
        return path
