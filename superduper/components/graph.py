import dataclasses as dc
import typing as t

import networkx as nx

from superduper.backends.base.query import Query
from superduper.backends.query_dataset import QueryDataset
from superduper.components.model import Model, Signature, ensure_initialized


def input_node(*args):
    """
    Create an IndexableNode for input.

    :param args: Arguments for the input.
    """
    return IndexableNode(
        model=Input(spec=args if len(args) > 1 else args[0]),
        parent_graph=nx.DiGraph(),
        parent_models={},
    )


def document_node(*args):
    """Create an IndexableNode for document input.

    :param args: Arguments for the document input.
    """
    return IndexableNode(
        model=DocumentInput(spec=args), parent_graph=nx.DiGraph(), parent_models={}
    )


class IndexableNode:
    """IndexableNode class to index the model.

    Create a model IndexableNode, which can be used to
    index the model while creating graph links.

    :param model: Model for the IndexableNode
    :param parent_graph: Parent graph
    :param parent_models: Parent models
    :param index: Index for arguments
    :param identifier: Identifier for the class
    """

    def __init__(
        self, *, model, parent_graph, parent_models, index=None, identifier=None
    ):
        self.model = model
        self.parent_graph = parent_graph
        self.parent_models = parent_models
        self.index = index
        self.identifier = identifier

    def __getitem__(self, item):
        """Method for indexing the model.

        :param item: Index
        """
        return IndexableNode(
            model=self.model,
            parent_graph=self.parent_graph,
            parent_models=self.parent_models,
            index=item,
            identifier=self.identifier,
        )

    def to_graph(self, identifier: str):
        """Helper method to get the graph form.

        :param identifier: Unique identifier
        """
        from superduper.components.graph import DocumentInput, Graph

        input_model = next(
            v
            for k, v in self.parent_models.items()
            if isinstance(self.parent_models[k].model, (Input, DocumentInput))
        )
        graph = Graph(
            identifier=identifier, input=input_model.model, outputs=[self.model]
        )
        for u, v, data in self.parent_graph.edges(data=True):
            u_node = self._get_node(u)
            v_node = self._get_node(v)
            graph.connect(u_node.model, v_node.model, on=(u_node.index, data['key']))
        return graph

    def _get_node(self, u):
        if u == self.model.identifier:
            return self
        return self.parent_models[u]

    def to_listeners(self, select: Query, identifier: str):
        """Create listeners from the parent graph and models.

        :param select: Query query
        :param identifier: Unique identifier
        """
        from superduper.components.application import Application
        from superduper.components.listener import Listener

        nodes = list(nx.topological_sort(self.parent_graph))
        input_node = next(
            v
            for k, v in self.parent_models.items()
            if isinstance(self.parent_models[k].model, DocumentInput)
        )
        assert isinstance(input_node.model, DocumentInput)
        listener_lookup: t.Dict = {}
        for node in nodes:
            in_edges = list(self.parent_graph.in_edges(node, data=True))
            if not in_edges:
                continue
            node = self._get_node(node)
            key: t.Dict = {}
            for u, _, data in in_edges:
                previous_node = self._get_node(u)
                if isinstance(previous_node.model, DocumentInput):
                    key[previous_node.index] = data['key']
                else:
                    assert previous_node.index is None
                    upstream_listener = listener_lookup[previous_node.model.identifier]
                    key[upstream_listener.outputs] = data['key']
            listener_lookup[node.model.identifier] = Listener(
                model=node.model,
                select=select,
                key=key,
                identifier=node.identifier,
            )
        return Application(
            identifier=identifier,
            components=list(listener_lookup.values()),
        )


class OutputWrapper:
    """OutputWrapper class for wrapping model outputs.

    :param r: Output
    :param keys: Output keys
    """

    def __init__(self, r, keys):
        self.keys = keys
        self.r = r

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.r[self.keys[item]]
        elif isinstance(item, str):
            return self.r[item]
        else:
            raise TypeError(f'Unsupported type for __getitem__: {type(item)}')


class Input(Model):
    """Root model of a graph.

    :param spec: Model specifications from `inspect`
    :param identifier: Unique identifier
    :param signature: Model signature
    """

    spec: t.Union[str, t.List[str]]
    identifier: str = '_input'
    signature: Signature = '*args'

    def __post_init__(self, db, artifacts, example):
        super().__post_init__(db, artifacts, example)
        if isinstance(self.spec, str):
            self.signature = 'singleton'

    def predict(self, *args):
        """Single prediction."""
        if self.signature == 'singleton':
            return args[0]
        return OutputWrapper(
            {k: arg for k, arg in zip(self.spec, args)}, keys=self.spec
        )

    def predict_batches(self, dataset):
        """Predict on the dataset.

        :param dataset: Series of datapoints
        """
        return [self.predict(dataset[i]) for i in range(len(dataset))]


class DocumentInput(Model):
    """Document Input node of the graph.

    :param spec: Model specifications from `inspect`
    :param identifier: Unique identifier
    """

    spec: t.Union[str, t.List[str]]
    identifier: str = '_input'
    signature: Signature = 'singleton'

    def __post_init__(self, db, artifacts, example):
        super().__post_init__(db, artifacts, example)

    def predict(self, r):
        """Single prediction.

        :param r: Model input
        """
        return {k: r[k] for k in self.spec}

    def predict_batches(self, dataset):
        """Predict on the dataset.

        :param dataset: Series of datapoints
        """
        return [self.predict(dataset[i]) for i in range(len(dataset))]


class Graph(Model):
    """Represents a directed acyclic graph composed of interconnected model nodes.

    This class enables the creation of complex predictive models
    by defining a computational graph structure where each node
    represents a predictive model. Models can be connected via edges
    to define dependencies and flow of data.

    The predict() method executes predictions through the graph, ensuring
    correct data flow and handling of dependencies.

    :param models: Graph model list.
    :param edges: Graph edges list.
    :param input: Graph root node.
    :param outputs: Graph output nodes.
    :param signature: Graph signature.

    Example:
    -------
    >> g = Graph(
    >>   identifier='simple-graph', input=model1, outputs=[model2], signature='*args'
    >> )
    >> g.connect(model1, model2)
    >> assert g.predict(1) == [(4, 2)]

    """

    _DEFAULT_ARG_WEIGHT: t.ClassVar[t.Tuple] = (None, 'singleton')
    type_id: t.ClassVar[str] = 'model'

    models: t.List[Model] = dc.field(default_factory=list)
    edges: t.List[t.Tuple[str, str, t.Tuple[t.Union[int, str], str]]] = dc.field(
        default_factory=list
    )
    input: Model
    outputs: t.List[t.Union[str, Model]] = dc.field(default_factory=list)
    signature: Signature = '*args,**kwargs'

    def __post_init__(self, db, artifacts, example):
        self.G = nx.DiGraph()
        self.nodes = {}
        self.version = 0
        self._db = None

        self.signature = self.input.signature
        if isinstance(self.outputs, list):
            self.output_identifiers = [
                o.identifier if isinstance(o, Model) else o for o in self.outputs
            ]
        else:
            self.output_identifiers = (
                self.outputs.identifier
                if isinstance(self.outputs, Model)
                else self.outputs
            )

        # Load the models and edges into a directed graph
        models = {m.identifier: m for m in self.models}
        if self.edges and models:
            for connection in self.edges:
                u, v, on = connection
                self.connect(
                    models[u],
                    models[v],
                    on=on,
                    update_edge=False,
                )
        super().__post_init__(db, artifacts=artifacts, example=example)

    def connect(
        self,
        u: Model,
        v: Model,
        on: t.Optional[t.Tuple[t.Union[int, str], str]] = None,
        update_edge: t.Optional[bool] = True,
    ):
        """Connect the relationship between two models.

        Connects two nodes `u` and `v` on an edge, where the edge is a tuple with
        the first element describing output index (int or None)
        and the second describing input argument (str).

        :param u: First node for connection.
        :param v: Second node for connection.
        :param on: Define incoming model output and connecting
                   model input connection.
        :param update_edge: Bool to update edge.

        Note:
        ----
        Output index: None means all outputs of node u are connected to node v.

        """
        assert isinstance(u, Model)
        assert isinstance(v, Model)

        if u.identifier not in self.nodes:
            self.nodes[u.identifier] = u
            self.G.add_node(u.identifier)

        if v.identifier not in self.nodes:
            self.nodes[v.identifier] = v
            self.G.add_node(v.identifier)

        G_ = self.G.copy()
        G_.add_edge(u.identifier, v.identifier, weight=on or self._DEFAULT_ARG_WEIGHT)
        if not nx.is_directed_acyclic_graph(G_):
            raise TypeError(
                'The graph is not a DAG with this'
                f'edge: {u.identifier} -> {v.identifier}'
            )
        self.G = G_

        if update_edge:
            self.edges.append(
                (u.identifier, v.identifier, on or self._DEFAULT_ARG_WEIGHT)
            )
            if isinstance(u, Model) and u not in self.models:
                self.models.append(u)
            if v not in self.models:
                self.models.append(v)
        return

    def fetch_output(
        self, output, index: t.Optional[t.Union[int, str]] = None, one: bool = False
    ):
        """Get corresponding output from model outputs with respect to link weight.

        :param output: model output.
        :param index: index to select output.
        :param one: if single prediction.
        """
        if index is not None:
            assert isinstance(index, (int, str))

            if isinstance(index, str):
                assert isinstance(
                    output, dict
                ), 'Output should be a dict for indexing with str'

            try:
                if one:
                    return output[index]
                else:
                    return [o[index] for o in output]

            except KeyError:
                raise KeyError("Model node does not have sufficient outputs")
        # Index None implies passing all outputs to dependent node.
        return output

    def validate(self, node):
        """Validates the graph for any disconnection.

        :param node: Graph node.
        """
        # TODO: Create a cache to reduce redundant validation in predict in db

        predecessors = list(self.G.predecessors(node))
        dependencies = [self.validate(node=p) for p in predecessors]
        model = self.nodes[node]
        if dependencies and len(model.inputs) != len(dependencies):
            raise TypeError(
                f'Graph disconnected at Node: {model.identifier} '
                f'and is partially connected with {dependencies}\n'
                f'Required connected node is {len(model.inputs)} '
                f'but got only {len(dependencies)}, '
                f'Node required params: {model.inputs.params}'
            )
        return node

    def _fetch_inputs(self, dataset, edges=[], outputs=[], node=None):
        arg_inputs = []
        kwargs = {}
        node_signature = self.nodes[node].signature

        def _length(lst):
            count = 0
            for item in lst:
                if isinstance(item, (list, tuple)):
                    count += _length(item)
                else:
                    count += 1
            return count

        if not _length(outputs):
            outputs = dataset

        for ix, edge in enumerate(edges):
            output_key, input_key = edge['weight']
            arg_input_dataset = self.fetch_output(outputs[ix], output_key, one=False)
            if input_key == 'singleton':
                # If singleton we override input_key with the actual model
                # singleton param.
                input_key = self.nodes[node].inputs.params[0]

            kwargs[input_key] = arg_input_dataset
            arg_inputs.append(arg_input_dataset)

        if not arg_inputs:
            return outputs

        batches = self._transpose(arg_inputs)
        mapped_dataset = []
        for batch in batches:
            batch = self._map_dataset_to_model(batch, node_signature, kwargs)
            mapped_dataset.append(batch)
        return mapped_dataset

    def _map_dataset_to_model(self, batch, signature, kwargs):
        if signature == '*args':
            return [[batch[p]] for p, _ in enumerate(kwargs)]
        elif signature == 'singleton':
            return batch[0]
        elif signature == '*args,**kwargs':
            return (), {k: batch[i] for i, k in enumerate(kwargs)}
        elif signature == '**kwargs':
            return {k: batch[i] for i, k in enumerate(kwargs)}

    def _fetch_input(self, args, kwargs, edges=[], outputs=[]):
        node_input = {}
        for ix, edge in enumerate(edges):
            output_key, input_key = edge['weight']

            node_input[input_key] = self.fetch_output(outputs[ix], output_key, one=True)
            kwargs = node_input
            args = ()

        if 'singleton' in node_input:
            args = (node_input['singleton'],)
            kwargs = {}
        return args, kwargs

    def _predict_on_node(self, *args, node=None, cache={}, one=True, **kwargs):
        if node not in cache:
            predecessors = list(self.G.predecessors(node))
            outputs = [
                self._predict_on_node(*args, **kwargs, one=one, node=p, cache=cache)
                for p in predecessors
            ]

            edges = [self.G.get_edge_data(p, node) for p in predecessors]

            if one is True:
                args, kwargs = self._fetch_input(
                    args, kwargs, edges=edges, outputs=outputs
                )
                cache[node] = self.nodes[node].predict(*args, **kwargs)
            else:
                dataset = self._fetch_inputs(
                    args[0], edges=edges, outputs=outputs, node=node
                )
                cache[node] = self.nodes[node].predict_batches(dataset=dataset)
            return cache[node]
        return cache[node]

    @ensure_initialized
    def predict(self, *args, **kwargs):
        """Predict on single data point.

        Single data point prediction passes the args and kwargs to the defined node flow
        in the graph.
        """
        # Validate the node for incompletion
        # TODO: Move to to_graph method and validate the graph
        # list(map(self.validate, self.output_identifiers))
        cache = {}

        if isinstance(self.output_identifiers, list):
            outputs = [
                self._predict_on_node(*args, node=output, cache=cache, **kwargs)
                for output in self.output_identifiers
            ]
            return outputs
        else:
            return self._predict_on_node(
                *args, node=self.output_identifiers, cache=cache, **kwargs
            )

    def patch_dataset_to_args(self, dataset):
        """Get the dataset and patch it with args.

        Patch the dataset with args type as default, since all
        corresponding nodes take args as input type.

        :param dataset: series of datapoint.
        """
        args_dataset = []
        signature = self.signature
        input_params = self.input.inputs

        def mapping(x, signature):
            if signature == '*args':
                return {p: d for d, p in zip(x, input_params.params)}
            elif signature == '*args,**kwargs':
                base_kwargs = x[1]
                kwargs = {a: d for a, d in zip(input_params.args, x[0])}
                return kwargs.update(base_kwargs)
            else:
                return x

        for data in dataset:
            data = mapping(data, signature)
            args_dataset.append(data)
        return args_dataset

    @ensure_initialized
    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """Predict on dataset i.e. series of datapoints.

        :param dataset: Series of datapoints.
        """
        # Validate the node for incompletion
        if isinstance(self.output_identifiers, list):
            list(map(self.validate, self.output_identifiers))
        else:
            self.validate(self.output_identifiers)

        if isinstance(dataset, QueryDataset):
            raise TypeError('QueryDataset is not supported in graph mode')
        cache: t.Dict[str, t.Any] = {}

        if isinstance(self.output_identifiers, list):
            outputs = [
                self._predict_on_node(dataset, node=output, cache=cache, one=False)
                for output in self.output_identifiers
            ]
        else:
            outputs = self._predict_on_node(
                dataset, node=self.output_identifiers, cache=cache, one=False
            )
        # TODO: check if output schema and datatype required
        return outputs

    @staticmethod
    def _transpose(outputs):
        transposed_outputs = []
        for i in range(len(outputs[0])):
            batch_outs = []
            for o in outputs:
                batch_outs.append(o[i])
            transposed_outputs.append(batch_outs)

        return transposed_outputs
