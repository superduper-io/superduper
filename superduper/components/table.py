import typing as t

from rich import print as rprint
from rich.tree import Tree

from superduper import CFG
from superduper.base.annotations import trigger
from superduper.base.schema import Schema
from superduper.components.component import Component
from superduper.misc import typing as st  # noqa: F401
from superduper.misc.importing import import_object

DEFAULT_PRIMARY_ID = 'id'

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class Table(Component):
    """
    A component that represents a table in a database.

    :param fields: The schema of the table
    :param primary_id: The primary id of the table
    :param data: Data to insert post creation
    :param path: The path to the class
    :param is_component: Whether the table is a component
    """

    fields: t.Dict | None = None
    primary_id: str = DEFAULT_PRIMARY_ID
    data: Component | None = None
    path: str | None = None
    is_component: bool = False
    component_cache: t.ClassVar[bool] = True

    def postinit(self):
        """Post initialization method."""
        if self.path is None:
            assert isinstance(self.fields, dict), "Fields must be set if cls is not set"
            self.schema = Schema.build(**self.fields)
            self.cls = None
        else:
            self.cls = import_object(self.path)
            self.schema = self.cls.class_schema
            self.fields = self.cls._new_fields
        super().postinit()

    def _build_outputs_graph(self, tree: Tree, event_type: str = 'insert'):
        """Show the outputs graph of the component."""
        assert (
            self.db is not None
        ), "Datalayer must be set before building outputs graph"
        db: 'Datalayer' = self.db
        components = db.metadata.show_cdcs(self.identifier)
        for component, identifier, uuid in components:
            cdc = db.load(component=component, identifier=identifier, uuid=uuid)
            methods = cdc.get_triggers(event_type=event_type)
            for method in methods:
                subtree = tree.add(
                    f'{cdc.huuid}.{method}',
                )
                tab: Table = db.load('Table', cdc.cdc_table)
                tab._build_outputs_graph(subtree, event_type=event_type)
        return tree

    def show_cdcs(self, event_type: str = 'insert'):
        """Show the CDCs of the table in tree format.

        :param event_type: The type of event to show the CDCs {insert, update, delete}
        """
        tree = Tree(self.identifier + f'[{event_type}]')
        self._build_outputs_graph(tree, event_type=event_type)
        rprint(tree)
        return

    def create_table_events(self):
        """Create the table events."""
        from superduper.base.event import CreateTable

        return {
            self.identifier: CreateTable(
                identifier=self.identifier,
                primary_id=self.primary_id,
                fields=self.fields,
                is_component=self.is_component,
            )
        }

    def cleanup(self):
        """Cleanup the table, on removal of the component."""
        if self.identifier.startswith(CFG.output_prefix):
            self.db.databackend.drop_table(self.identifier)

    @trigger('apply', requires='data')
    def add_data(self):
        if isinstance(self.data, Component):
            data = self.data.data
        else:
            data = self.data
        if data:
            self.db[self.identifier].insert(data)
