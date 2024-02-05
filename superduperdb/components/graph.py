import dataclasses as dc
import typing as t
from collections import OrderedDict
from functools import cached_property, partial
from inspect import signature

import networkx as nx

from superduperdb.base.artifact import Artifact
from superduperdb.components.component import Component
from superduperdb.components.model import Model, _Predictor
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
        self.encoder = model.encoder
        self.output_schema = model.output_schema

        predict_method = model.predict_method
        if isinstance(model, TorchModel):
            predict_method = model.forward_method

        assert isinstance(model.object, Artifact)
        self.object = model.object.artifact
        self.model = model
        self._generic_model = type(model) == Model

        if predict_method:
            self.predict_method = getattr(self.object, predict_method)
        else:
            self.predict_method = self.object

        self.predict_kwargs = model.predict_kwargs or {}
        self.predict_method = partial(self.predict_method, **self.predict_kwargs)

        # TODO: CAUTION: check if this is not override model.
        if predict_method:
            setattr(self.object, predict_method, self.predict_method)
        else:
            self.object = self.predict_method

        params = self.set_input()
        self.input = KeyStore(model.identifier, params)

    def set_input(self):
        sig = signature(self.predict_method)
        sig_keys = list(sig.parameters.keys())
        params = []
        for k in sig_keys:
            if k in self.predict_kwargs or (
                k == 'kwargs' and sig.parameters[k].kind == 4
            ):
                continue
            params.append(k)
        return params

    def predict(self, *args, one=False, **kwargs):
        if self._generic_model:
            out = self._forward(*args, one=one, **kwargs)
        else:
            out = self.model._predict(*args, one=one, **kwargs)
        return out

    def _predict(self, *args, **kwargs):
        outputs = self.predict_method(*args, **kwargs)
        return outputs

    def _forward(
        self, *args, one=False, num_workers: int = 0, **kwargs
    ) -> t.Sequence[int]:
        if self.model.batch_predict or one is True:
            return self._predict(*args, **kwargs)

        if args and kwargs:
            raise ValueError(
                'Args and Kwargs at the same time not supported in graph mode'
            )
        if args:
            return self._mutli_forward_args(*args, num_workers=num_workers)
        return self._mutli_forward_kargs(num_workers=num_workers, **kwargs)

    def _mutli_forward_kargs(self, num_workers=1, **kwargs):
        outputs = []
        if num_workers:
            import multiprocessing

            pool = multiprocessing.Pool(processes=num_workers)
            for r in pool.starmap(self._predict, zip(*list(kwargs.values()))):
                outputs.append(r)
            pool.close()
            pool.join()
        else:
            for r in zip(*list(kwargs.values())):
                outputs.append(self._predict(*r))
        return outputs

    def _mutli_forward_args(self, *args, num_workers=1):
        outputs = []
        if num_workers:
            to_call = partial(self._predict, *args)
            import multiprocessing

            pool = multiprocessing.Pool(processes=num_workers)
            for r in pool.map(to_call, zip(*args)):
                outputs.append(r)
            pool.close()
            pool.join()
        else:
            for r in zip(*args):
                outputs.append(self._predict(*r))
        return outputs


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
        self._output = output

    def predict(self, *args, **kwargs):
        self.output = self.model.predict(*args, **kwargs)
        return self.output


@dc.dataclass(kw_only=True)
class Graph(Component, _Predictor):
    db: 'Datalayer'
    _DEFAULT_ARG_WEIGHT: t.ClassVar[str] = '_base'
    _input: t.ClassVar[str] = '_Graph_input'

    def __post_init__(self):
        self.G = nx.DiGraph()
        self._key_store = {}
        self.nodes = {}
        self._node_output_cache = {}
        self.version = 0

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

    def _predict(self, X: t.Any, one: bool = False, **predict_kwargs):
        if self._input not in self.G.nodes:
            raise TypeError(
                'Root graph node is not present'
                ', make sure to add graph node'
                'with atleast one other node'
            )
        path = self.traversal(self.G, [self._input], [self._input])
        self.validate(path)
        output = None

        # Update graph encoder and outschema as per output node.
        output_node = self.nodes[path[-1]]
        self.encoder = output_node.model.encoder
        self.output_schema = output_node.model.output_schema

        for graph_node in path:
            node = self.nodes[graph_node]

            if graph_node == '_Graph_input':
                node.output = X
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
            output = node.predict(*args, one=one, **kwargs)

        return output

    def show(self):
        path = self.traversal(self.G, [self._input], [self._input])
        path = ' --> '.join(path)
        print(path)
        return path
