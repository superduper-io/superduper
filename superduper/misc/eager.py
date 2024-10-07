import dataclasses as dc
import typing as t
from enum import Enum

import networkx as nx

from superduper import CFG, logging
from superduper.base.constant import KEY_BLOBS, KEY_BUILDS, KEY_FILES
from superduper.base.leaf import build_uuid

if t.TYPE_CHECKING:
    from superduper.backends.base.query import Query
    from superduper.components.model import Model

_MIXED_FLATTEN_ERROR_MESSAGE = (
    "Detected a flattened storage type in the upstream model.\n"
    "The current eager mode does not support mixed dependencies on "
    "flattened and non-flattened upstream models, meaning the downstream "
    "task can only depend on a single flattened model or multiple non-flattened models."
)


class SuperDuperDataType(str, Enum):
    """The SuperDuper data type enum."""

    DATA = "data"
    MODEL_OUTPUT = "model_output"
    CONSTANT = "constant"


@dc.dataclass
class SuperDuperData:
    """A data class that represents the data in eager mode.

    :param data: The real data to be stored.
    :param type: The type of the SuperDuperData. ["data", "model_output", "constant"]
    :param graph: The graph object, used to build the graph.
    :param ops: The operations performed on the data. Used to build the key.
    :param query: The query object. Only required for the root node (The data node).
    :param model: The model object. Only required for the model output node.
    :param source: The source object. Used to track the original node.
    """

    data: t.Any
    type: SuperDuperDataType
    graph: t.Optional["Graph"] = None
    ops: t.List[t.Union[int, str]] = dc.field(default_factory=list)
    query: t.Optional["Query"] = None
    model: t.Optional["Model"] = None
    source: t.Optional["SuperDuperData"] = None
    flatten: bool = False

    def __post_init__(self):
        if self.type == SuperDuperDataType.DATA and self.graph is None:
            Graph(self)

        self._children_cache = {}
        self._predict_id = None
        self._filter = {}
        self._filter_nodes = set()
        self._listener_kwargs = {}
        self.source = self.source or self
        self._compare = None

    def apply(self):
        """Apply the node to the graph."""
        self.graph.apply_nodes(self.source)

    @property
    def predict_id(self):
        """The predict id of the model output node."""
        if self._predict_id is None:
            assert self.model is not None, "Model is required."
            self._predict_id = self.model.identifier + '__' + build_uuid()

        return self._predict_id

    @predict_id.setter
    def predict_id(self, value):
        """Set the predict id of the model output node."""
        assert isinstance(value, str), "Predict id must be a string."
        self._predict_id = value
        if self is not self.source:
            self.source._predict_id = value

    @property
    def listener_kwargs(self):
        """The listener kwargs of the model output node."""
        return self._listener_kwargs

    @listener_kwargs.setter
    def listener_kwargs(self, value):
        """Set the listener kwargs of the model output node."""
        assert isinstance(value, dict), "Listener kwargs must be a dictionary."
        self._listener_kwargs = value

    def __getitem__(self, item):
        if item in self._children_cache:
            return self._children_cache[item]
        new_sdd = self.copy()
        new_sdd.data = self.data[item]
        new_sdd.ops.append(item)
        self._children_cache[item] = new_sdd
        return new_sdd

    def __str__(self):
        if self._compare:
            return self._compare_string()
        return str(self.data)

    def __repr__(self):
        if self._compare:
            return self._compare_string()
        return repr(self.data)

    def _compare_string(self):
        compare_string = "!=" if self._compare[0] == "eq" else "=="
        return f"{self.key} {compare_string} {self._compare[1]}"

    @property
    def dict(self):
        """Return the SuperDuperData object as a dictionary."""
        return dc.asdict(self)

    @staticmethod
    def detect_and_get_graph(*args, **kwargs) -> t.Tuple[bool, t.Union["Graph", None]]:
        """Detect and get the graph object from the arguments."""
        sdd_data = [
            var for var in [*args, *kwargs.items()] if isinstance(var, SuperDuperData)
        ]

        graph = sdd_data[0].graph if sdd_data else None

        return bool(graph), graph

    def __hash__(self):
        return id(self)

    def copy(self):
        """Copy the SuperDuperData object."""
        new_sdd = SuperDuperData(
            self.data,
            self.type,
            ops=self.ops.copy(),
            graph=self.graph,
            query=self.query,
            model=self.model,
            source=self.source,
        )
        if self.model:
            new_sdd.predict_id = self.predict_id
        return new_sdd

    @property
    def key(self):
        """Build the key for the node.

        The key is used to build the downstream listener key.
        """
        assert all(
            isinstance(op, str) for op in self.ops
        ), "Only support string type operations for building key"

        if self.type == SuperDuperDataType.DATA:
            key = ".".join(self.ops)
        elif self.type == SuperDuperDataType.MODEL_OUTPUT:
            prefix = f"{CFG.output_prefix}{self.predict_id}"
            if self.ops:
                key = f"{prefix}.{'.'.join(self.ops)}"
            else:
                key = prefix
        else:
            raise ValueError(f"Unknown node type: {self.type}")

        return key

    def __eq__(self, other):
        if isinstance(other, SuperDuperData):
            return self.data == other.data

        new_sdd = self.copy()
        new_sdd._compare = ("eq", other)
        return new_sdd

    def __ne__(self, other):
        if isinstance(other, SuperDuperData):
            return self.data != other.data

        new_sdd = self.copy()
        new_sdd._compare = ("ne", other)
        return new_sdd

    def set_condition(self, data: "SuperDuperData"):
        """Set the condition for the query."""
        if not data.ops and data.ops[-1] not in ("eq", "ne"):
            raise ValueError("Condition must be set after a comparison operation")
        self._filter[data.key] = data._compare
        self._filter_nodes.add(data.source)

    @property
    def filter(self):
        """Return the filter for building the query."""
        return self._filter

    @property
    def filter_nodes(self):
        """Return the filter nodes for building the query."""
        return self._filter_nodes


class Graph:
    """The graph object used to build the graph.

    :param root: The root node of the graph. Must be a data node.
    """

    def __init__(self, root: SuperDuperData):
        self._graph = nx.DiGraph()
        self.root = root
        self.root.graph = self
        self._apply_cache: t.Set[str] = set()

    @property
    def db(self):
        """The database object."""
        return self.root.query.db

    def add_edge(
        self,
        upstream: SuperDuperData,
        downstream: SuperDuperData,
        track_data: t.List["TrackData"],
    ):
        """Add an edge to the graph.

        :param upstream: The upstream node.
        :param downstream: The downstream node.
        :param track_data: The track data object.
        """
        self._graph.add_edge(upstream, downstream, track_data=track_data)

    def _find_upstream_nodes(self, node: SuperDuperData):
        """Returns all upstream nodes of the given node."""
        return list(nx.ancestors(self._graph, node))

    def _find_upstream_nodes_edges(self, node: SuperDuperData):
        """Returns all incoming edges for a given node.

        This function will flatten the track data.

        :param node: The node to find the incoming edges.
        """
        relations = []
        for upstream, downstream, track_data in self._graph.in_edges(
            node, data="track_data"
        ):
            for track in track_data:
                relations.append((upstream, downstream, track))
        return relations

    def _find_nodes_for_apply(self, *nodes):
        topo_sort_nodes = list(nx.topological_sort(self._graph))

        sort_hash = {node: i for i, node in enumerate(topo_sort_nodes)}

        apply_nodes = set(nodes)
        for node in nodes:
            # Upstream nodes are used to query data from upstream sources.
            upstream_nodes = self._find_upstream_nodes(node)
            apply_nodes.update(upstream_nodes)

            # filter_nodes are used to filter the upstream data
            # Don't need to query the the data from upstream filter nodes
            apply_nodes.update(node.filter_nodes)

        apply_nodes = sorted(apply_nodes, key=lambda x: sort_hash[x])

        apply_nodes = [
            node for node in apply_nodes if node.type == SuperDuperDataType.MODEL_OUTPUT
        ]

        return apply_nodes

    def apply_nodes(self, *nodes):
        """Apply the nodes to the graph.

        :param nodes: The nodes to apply.
        """
        apply_nodes = self._find_nodes_for_apply(*nodes)
        predict_ids = [node.predict_id for node in apply_nodes]
        logging.info(f"Applying nodes: {predict_ids}")
        for node in apply_nodes:
            self._apply_node(node)

    def _apply_node(self, node: SuperDuperData):
        if node.type != SuperDuperDataType.MODEL_OUTPUT:
            return

        if node.predict_id in self._apply_cache:
            logging.info(f"Node [{node.predict_id}] already applied.")
            return

        logging.info(f"Applying node: {node.predict_id}")
        logging.info(f"Example output {node.data}")

        assert node.source is not None, "Source is required."
        key = self._get_key(node.source)
        logging.info(f"Key: {key}")
        assert key, "Key is required."
        select = self._get_select(node.source)
        logging.info(f"Select: {select}")
        predict_id = node.predict_id
        logging.info(f"Predict id: {predict_id}")
        predict_kwargs = node.listener_kwargs
        logging.info(f"Predict kwargs: {predict_kwargs}")

        assert node.model is not None, "Model is required."
        listener = node.model.to_listener(
            key=key,
            select=select,
            identifier=predict_id.split('__')[0],
            uuid=predict_id.split('__')[1],
            predict_kwargs=predict_kwargs,
            flatten=node.flatten,
        )
        logging.info(f"Listener: {listener}")

        self.db.apply(listener)
        self._apply_cache.add(node.predict_id)

    def _get_key(self, node: SuperDuperData):
        relations = self._find_upstream_nodes_edges(node)
        arg_relations = [
            relation for relation in relations if isinstance(relation[2].key, int)
        ]
        if arg_relations and len(arg_relations) - 1 != max(
            map(lambda x: x[2].key, arg_relations)
        ):
            raise ValueError(
                f"Arguments error, {list(map(lambda x: x[2], arg_relations))}"
            )

        kwarg_relations = [
            relation for relation in relations if isinstance(relation[2].key, str)
        ]

        args_keys = []
        for relation in arg_relations:
            upstream_node, _, track_data = relation
            args_keys.append((track_data.key, track_data.value.key))

        args_keys = tuple([key for _, key in sorted(args_keys, key=lambda x: x[0])])

        kwargs_keys = {}
        for relation in kwarg_relations:
            upstream_node, _, track_data = relation
            kwargs_keys[track_data.key] = track_data.value.key

        key = args_keys, kwargs_keys
        if not kwargs_keys:
            if len(args_keys) == 1:
                key = args_keys[0]
            else:
                key = args_keys

        elif not args_keys:
            key = kwargs_keys

        return key

    def _get_select(self, node: SuperDuperData):
        relations = self._find_upstream_nodes_edges(node)

        assert self.root.query, "Query is required."
        main_table = root_table = self.root.query.table
        main_table_keys = []
        predict_ids = []

        for relation in relations:
            upstream_node, _, track_data = relation

            if upstream_node.type == SuperDuperDataType.DATA:
                assert (
                    root_table == upstream_node.query.table
                ), "Data node must have the same table as the root node."
                main_table_keys.append(track_data.value.key)
                continue

            elif upstream_node.type == SuperDuperDataType.MODEL_OUTPUT:
                if not upstream_node.flatten:
                    predict_ids.append(upstream_node.predict_id)
                else:
                    if len(relations) != 1:
                        raise ValueError(_MIXED_FLATTEN_ERROR_MESSAGE)
                    main_table = f"{CFG.output_prefix}{upstream_node.predict_id}"
                    predict_ids = []
            else:
                raise ValueError(f"Unknown node type: {upstream_node.type}")

        if main_table != root_table:
            select = self.db[main_table].select()

        else:
            from superduper.base.enums import DBType

            if self.db.databackend.db_type == DBType.MONGODB:
                if main_table_keys:
                    main_table_keys.extend(
                        [KEY_BUILDS, KEY_FILES, KEY_BLOBS, "_schema"]
                    )
                select = self.db[main_table].find({}, {k: 1 for k in main_table_keys})

            else:
                if "id" not in main_table_keys:
                    main_table_keys.insert(0, "id")
                select = self.db[main_table].select(*main_table_keys)

            if node.filter:
                for key, value in node.filter.items():
                    if value[0] == "ne":
                        select = select.filter(select[key] != value[1])
                    else:
                        select = select.filter(select[key] == value[1])

            if predict_ids:
                select = select.outputs(*predict_ids)

        return select


class TrackData:
    """The track data object used to track the data in the graph.

    :param key: The key of the track data.
    :param value: The value of the track data.
    """

    def __init__(self, key, value: SuperDuperData):
        self.key = key
        self.value = value

    def __str__(self):
        return f"{self.key} -> {self.value.key}"

    __repr__ = __str__
