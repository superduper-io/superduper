import dataclasses as dc
import typing as t
from collections import OrderedDict

import networkx as nx

from superduperdb.components.component import Component
from superduperdb.components.model import Model, _Predictor


class GraphModel:
    def __init__(self, name):
        self.identifier = name
        self.inputs = []

    def predict(self, x):
        return x


class IntermediateGraphNode:
    def __init__(self, node, graph):
        self.node = node
        self.G = graph

    @property
    def output(self):
        return self.G.get_output(self.node)


@dc.dataclass(kw_only=True)
class Graph(Component, _Predictor):
    models: t.List[Model] = dc.field(default_factory=list)
    edges: t.List[t.Tuple[str, str, t.Union[None, str]]] = dc.field(
        default_factory=list
    )
    _DEFAULT_ARG_WEIGHT: t.ClassVar[str] = '_base'
    type_id: t.ClassVar[str] = 'graph'

    def __post_init__(self, artifacts):
        self.G = nx.DiGraph()
        self.nodes = {}
        self._node_output_cache = {}
        self.version = 0
        self._db = None

        # Load the models and edges into a di graph
        models = {m.identifier: m for m in self.models}
        if self.edges and models:
            for connection in self.edges:
                u, v, on = connection
                self.connect(
                    models[u] if u != self.identifier else self,
                    models[v],
                    on=on,
                    update_edge=False,
                )
        super().__post_init__(artifacts=artifacts)

    def connect(
        self,
        u: Component,
        v: Model,
        on: t.Optional[str] = None,
        update_edge: t.Optional[bool] = True,
    ):
        assert isinstance(u, (Model, Graph))
        assert isinstance(v, Model)

        if u.identifier not in self.nodes:
            if isinstance(u, Graph):
                self.nodes[u.identifier] = GraphModel(self.identifier)
            else:
                self.nodes[u.identifier] = u
            self.G.add_node(u.identifier)

        if v.identifier not in self.nodes:
            self.nodes[v.identifier] = v
            self.G.add_node(v.identifier)

        G_ = self.G.copy()
        G_.add_edge(u.identifier, v.identifier, weight=on or self._DEFAULT_ARG_WEIGHT)

        if not nx.is_directed_acyclic_graph(G_):
            raise TypeError('The graph is not DAG with this edge')
        self.G = G_

        if update_edge:
            self.edges.append(
                (u.identifier, v.identifier, on or self._DEFAULT_ARG_WEIGHT)
            )
            if isinstance(u, Model) and u not in self.models:
                self.models.append(u)
            if v not in self.models:
                self.models.append(v)
        return IntermediateGraphNode(self.nodes[v.identifier], self)

    def stash_node_output(self, node, output):
        self._node_output_cache[node.identifier] = output

    def get_output(self, node):
        return self._node_output_cache[node.identifier]

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

            if len(node.inputs) != len(arg_nodes):
                raise TypeError(
                    f'Graph disconnected at Node: {node.identifier} '
                    f'and is partially connected with {nodes}\n'
                    f'Required connected node is {len(node.inputs)} '
                    f'but got only {len(nodes)}, '
                    f'Node required params: {node.inputs.params}'
                )

    def _predict(self, X: t.Any, one: bool = False, **predict_kwargs):
        if self.identifier not in self.G.nodes:
            raise TypeError(
                'Root graph node is not present'
                ', make sure to add graph node'
                'with atleast one other node'
            )
        path = self.traversal(self.G, [self.identifier], [self.identifier])
        self.validate(path)
        output = None

        # Update graph datatype and outschema as per output node.
        output_node = self.nodes[path[-1]]
        self.datatype = output_node.datatype
        self.output_schema = output_node.output_schema

        for graph_node in path:
            node = self.nodes[graph_node]

            if graph_node == self.identifier:
                self.stash_node_output(node, X)
                continue

            predecessors = list(self.G.predecessors(graph_node))

            arg_nodes = list(map(lambda x: self.nodes[x], predecessors))
            node_input = {}

            for predecessor, arg in zip(predecessors, arg_nodes):
                data = self.G.get_edge_data(predecessor, graph_node)
                key = data['weight']

                node_input[key] = self.get_output(arg)

            if self._DEFAULT_ARG_WEIGHT in node_input:
                node_input = node_input['_base']
            output = node.predict(node_input, one=one)
            self.stash_node_output(node, output)

        return output

    def show(self):
        path = self.traversal(self.G, [self.identifier], [self.identifier])
        path = ' --> '.join(path)
        print(path)
        return path
