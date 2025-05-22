import typing as t

import networkx as nx

from superduper import CFG, Component

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class CDC(Component):
    """Trigger actions when new data arrives in a table.

    :param cdc_table: Table which fires the triggers.
    """

    triggers: t.ClassVar[t.Set] = set()
    cdc_table: str
    services: t.ClassVar[t.Sequence[str]] = ('scheduler', 'cdc')

    @property
    def dependent_tables(self):
        """Get tables of this component."""
        return [self.cdc_table]

    def handle_update_or_same(self, other):
        """Handle the case in which the component is update without breaking changes.

        :param other: The other component to handle.
        """
        super().handle_update_or_same(other)
        other.cdc_table = self.cdc_table

    def cleanup(self):
        """Cleanup the component."""
        super().cleanup()
        self.db.cluster.cdc.drop_component(self.component, self.identifier)
        self.db.cluster.scheduler.drop_component(self.component, self.identifier)


def _get_cdcs_on_table(table, db: 'Datalayer'):
    from superduper.components.listener import Listener

    cdcs = db.metadata.show_cdcs(table)
    out = []
    for r in cdcs:
        component = db.load(component=r['component'], uuid=r['uuid'])
        out.append(component)
    return out


def _get_upstream_cdc_components(component: CDC, db: 'Datalayer'):
    out = []
    for upstream in component.get_children():
        if hasattr(upstream, 'cdc_table'):
            out.append(upstream)
    return out


def _get_downstream_cdc_components(component: CDC, db: 'Datalayer'):
    parents = db.metadata.get_component_version_parents(component.uuid)
    out = []
    for parent_component, parent_uuid in parents:
        r = db.metadata.get_component_by_uuid(parent_component, parent_uuid)
        if r.get('cdc_table'):
            out.append(db.load(parent_component, uuid=parent_uuid))
    return out


def build_streaming_graph(table, db: "Datalayer") -> nx.DiGraph:
    """Build a streaming graph from a table.

    The graph has as each node a component which
    ingests from the table, or ingests from
    a component which ingests from the table (etc.).

    This function constructs a directed graph representing the data flow
    between components connected to the specified table.

    :param table: The table to build the graph from.
    :param db: Datalayer instance
    :return: A directed graph (DiGraph) representing the streaming components network
    """
    # Initialize an empty directed graph
    G = nx.DiGraph()

    # Get all components directly connected to the specified table
    components = _get_cdcs_on_table(table, db)
    initial_components_set = {c.huuid for c in components}

    # Add these components as nodes and their upstream dependencies as edges
    for component in components:
        # Add the component as a node with its attributes
        G.add_node(component.huuid, component=component)

        # Find and add any upstream components that feed into this component
        for upstream in _get_upstream_cdc_components(component, db=db):
            # Skip if the upstream component is not in the initial set
            if upstream.huuid not in initial_components_set:
                continue
            # Create an edge representing data flow from upstream to current component
            G.add_edge(upstream.huuid, component.huuid)

    # Track which components we've already processed to avoid cycles
    visited = set()

    # Breadth-first traversal to discover all downstream components
    while components:
        new = []  # List to collect the next level of components

        for component in components:
            # Skip if we've already processed this component
            if component.huuid in visited:
                continue

            # Mark as visited
            visited.add(component.huuid)

            # Get all components that consume data from this component
            downstreams = _get_downstream_cdc_components(component, db=db)

            # Add each downstream component and connection to the graph
            for downstream in downstreams:
                G.add_node(downstream.huuid, component=downstream)
                G.add_edge(component.huuid, downstream.huuid)

            # Add downstream components to be processed in the next iteration
            new.extend(list(downstreams))

        # Deduplicate components for the next iteration (using dict comprehension)
        components = list({x.huuid: x for x in new}.values())

    # Return the completed streaming graph
    return G
