import networkx as nx
import typing as t

from superduper import Component, CFG

if t.TYPE_CHECKING:
    pass


class CDC(Component):
    """Trigger a function when a condition is met.

    ***Note that this feature deploys on superduper.io Enterprise.***

    :param cdc_table: Table which fires the triggers.
    """

    triggers: t.ClassVar[t.Set] = set()
    type_id: t.ClassVar[str] = 'cdc'
    cdc_table: str

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)

    def declare_component(self, cluster):
        super().declare_component(cluster)
        self.db.cluster.queue.put(self)
        self.db.cluster.cdc.put(self)

    @property
    def dependencies(self):
        """Get dependencies of this component."""
        if self.cdc_table.startswith(CFG.outputs_prefix):
            return [tuple(self.cdc_table.split('__'))]
        return []


def _get_parent_cdcs_of_component(component, db: 'Datalayer'):
    parents = db.metadata.get_component_version_parents(component.uuid)
    out = []
    for uuid in parents:
        r = db.metadata.get_component_by_uuid(uuid)
        if r.get('cdc_table'):
            out.append(db.load(uuid=uuid))
    return {c.uuid: c for c in out}


def _get_cdcs_on_table(table, db: 'Datalayer'):
    from superduper.components.listener import Listener
    cdcs = db.metadata.show_cdcs(table)
    out = []
    for uuid in cdcs:
        component = db.load(uuid=uuid)
        if isinstance(component, Listener):
            if len(component.select.tables) > 1:
                continue
            out.append(component)
        out.append(component)
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
                deps = [x[1] for x in parents[parent].dependencies]
                if component.uuid not in deps:
                    continue
                G.add_edge(component.huuid, parents[parent].huuid)
                out_cache[parents[parent].huuid] = parents[parent]
            new.extend(list(parents.values()))
        components = list({x.huuid: x for x in new}.values())
    return G, out_cache


if __name__ == '__main__':
    from superduper.components.model import ObjectModel
    from superduper import Listener
    from superduper import superduper

    db = superduper('mongomock://test')

    db['documents'].insert([{'x': i} for i in range(10)]).execute()
    
    m = ObjectModel('test', object=lambda x: x)
    m2 = ObjectModel('test', object=lambda x, y: x)

    l1 = Listener('l1', model=m, select=db['documents'].select(), key='x')

    db.apply(l1)

    l2 = Listener('l2', model=m, key=l1.outputs, select=l1.outputs_select)

    db.apply(l2)

    l3 = Listener('l3', model=m, select=db['documents'].select(), key='x')

    db.apply(l3)

    l4 = Listener('l4', model=m2, select=db['documents'].select().outputs(l1.predict_id, l3.predict_id), key=(l1.outputs, l3.outputs))

    db.apply(l4)

    G, components = build_streaming_graph('documents', db)

    def iterate_upwards(graph, start_nodes):
        visited = set()
        nodes_to_explore = list(start_nodes)

        while nodes_to_explore:
            current_node = nodes_to_explore.pop()
            
            for parent in graph.successors(current_node):
                if parent not in visited:
                    yield current_node, parent
                    visited.add(parent)
                    nodes_to_explore.append(parent)

    root_nodes = [n for n, d in G.in_degree() if d == 0]
    print(root_nodes)

    for child, parent in iterate_upwards(G, root_nodes):
        print(f"Step: Moving from {child} to {parent}")
