from inspect import signature

import networkx as nx

from superduperdb.components.component import Component
from superduperdb.components.model import Model


class Key(str):
    ...


class KeyStore:
    def __init__(self, model, params):
        self.model = model
        self._params = params
        for param in params:
            setattr(self, param, f'{model}.{param}')


class GraphModel:
    def __init__(self, model: 'Model'):
        self.identifier = model.identifier
        self.output = f'{model.identifier}.output'
        self.forward_method = model.forward_method
        self.object = model.object.artifact

        params = self.set_input(model.object.artifact)
        self.input = KeyStore(model.identifier, params)

    def set_input(self, object):
        predict = getattr(object, self.forward_method)

        sig = signature(predict)
        return list(sig.parameters.keys())


class Node:
    @staticmethod
    def get_key(name):
        key = '_base'
        if name == '_Graph_input':
            key = '_base'
        else:
            keys = name.split('.')
            if len(keys) == 1:
                key = '_base'
            else:
                key = keys[-1]
        return key

    def __init__(self, name, db=None):
        self.name = name.split('.')[0]
        key = name.split('.')[-1]
        self.key = key if key else '_base'

        self.model = GraphModel(db.load(name))
        self._output = None

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, output):
        self._output = output


class Graph(Component):
    db: 'Datalayer'

    def __post_init__(self):
        self.input = '_Graph_input'
        self.G = nx.DiGraph()
        self._key_store = {}
        self.nodes = {}
        self._node_output_cache = {}

    def add_edge(self, u, v):
        v_key = None

        if isinstance(v, Model):
            v_key = '_base'
            v = f'{v.identifier}.{v_key}'

        elif isinstance(u, Model):
            u = u.identifier

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
            v_key = v_key if v_key else node.key

        self.G.add_edge(u, v, weight=v_key)

    def stash_node_output(self, node, output):
        self._node_output_cache[node.name] = output

    def group_nodes_by_degree(self, nodes):
        from collections import OrderedDict

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
            return traversal_path.append(nodes[0])
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
        S = graph.subgraph(neighbors)
        traversal_path = self.level_traversal(S, neighbors, traversal_path)

        neighbors = find_level_neighbors(graph, neighbors)
        return self.traversal(graph, neighbors, traversal_path)

    def predict(self, x, one=True, select=None):
        # TODO: Implement select logic
        path = self.traversal(self.G, [self.input], ['_Graph_input'])
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
            if '_base' in kwargs:
                args = list(kwargs.values())
                kwargs = {}
            model = getattr(node.model.object, node.model.forward_method)
            output = model(*args, **kwargs)
            node.output = output

        return output

    def show(self):
        path = self.traversal(self.G, [self.input], ['_Graph_input'])
        path = ' --> '.join(path)
        print(path)
        return path
