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

    def handle_update_or_same(self, other):
        """Handle the case in which the component is update without breaking changes.

        :param other: The other component to handle.
        """
        super().handle_update_or_same(other)
        other.cdc_table = self.cdc_table

    def on_create(self):
        """Declare the component to the cluster."""
        super().on_create()
        self.db.cluster.scheduler.put_component(self)
        self.db.cluster.cdc.put_component(self)

    @property
    def dependencies(self):
        """Get dependencies of this component."""
        if self.cdc_table.startswith(CFG.output_prefix):
            return [tuple(['Listener'] + list(self.cdc_table.split('__')[1:]))]
        return []


def _get_parent_cdcs_of_component(component, db: 'Datalayer'):
    parents = db.metadata.get_component_version_parents(component.uuid)
    out = []
    for parent_component, parent_uuid in parents:
        r = db.metadata.get_component_by_uuid(parent_component, parent_uuid)
        if r.get('cdc_table'):
            out.append(db.load(parent_component, uuid=parent_uuid))
    return {c.uuid: c for c in out}


def _get_cdcs_on_table(table, db: 'Datalayer'):
    from superduper.components.listener import Listener

    cdcs = db.metadata.show_cdcs(table)
    out = []
    for r in cdcs:
        component = db.load(component=r['component'], uuid=r['uuid'])
        if isinstance(component, Listener) and component.select is not None:
            if len(component.select.tables) > 1:
                continue
            out.append(component)
            continue
        out.append(component)  # type: ignore[arg-type]
    return out


def build_streaming_graph(table, db: 'Datalayer'):
    """Build a streaming graph from a table.

    The graph has as each node a component which
    ingests from the table, or ingests from
    a component which ingests from the table (etc.).

    :param table: The table to build the graph from.
    :param db: Datalayer instance
    """
    G = nx.DiGraph()
    components = _get_cdcs_on_table(table, db)
    out_cache = {component.huuid: component for component in components}
    for component in components:
        G.add_node(component.huuid)
    while components:
        new = []
        for component in components:
            parents = _get_parent_cdcs_of_component(component, db=db)
            for parent in parents:
                deps = [x[-1] for x in parents[parent].dependencies]
                if component.uuid not in deps:
                    continue
                G.add_edge(component.huuid, parents[parent].huuid)
                out_cache[parents[parent].huuid] = parents[parent]
            new.extend(list(parents.values()))
        components = list({x.huuid: x for x in new}.values())
    return G, out_cache
