import typing as t

import networkx

from superduper import CFG, logging

from .component import Component

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class Application(Component):
    """
    A placeholder to hold list of components with associated funcionality.

    Components are sorted in a way that respects their mutual dependencies.

    :param components: List of components to group together and apply to `superduper`.
    :param namespace: List of tuples with type_id and identifier of components to
        assist in managing application.
    :param link: A reference link to web app serving the application
                 i.e. streamlit, gradio, etc
    """

    breaks: t.ClassVar[t.Sequence[str]] = ('components',)

    literals: t.ClassVar[t.Sequence[str]] = ("template",)
    type_id: t.ClassVar[str] = "application"
    components: t.Sequence[Component]
    namespace: t.Optional[t.Sequence[t.Tuple[str, str]]] = None
    link: t.Optional[str] = None

    def __post_init__(self, db):
        super().__post_init__(db)
        self._sort_components_and_set_upstream()

    def _sort_components_and_set_upstream(self):
        logging.info('Resorting components based on topological order.')
        G = networkx.DiGraph()
        lookup = {c.huuid: c for c in self.components}
        for k in lookup:
            G.add_node(k)
            for d in lookup[k].get_children_refs():  # dependencies:
                if d in lookup:
                    G.add_edge(d, k)
                    if lookup[d].upstream is None:
                        lookup[d].upstream = []

        for e in G.edges:
            if lookup[e[1]].upstream is None:
                lookup[e[1]].upstream = []
            lookup[e[1]].upstream.append(lookup[e[0]])

        nodes = list(networkx.topological_sort(G))
        logging.info(f'New order of components: {nodes}')
        components = [lookup[n] for n in nodes]
        self.components = components

    def pre_create(self, db: "Datalayer"):
        """Pre-create hook.

        :param db: Datalayer instance
        """
        self.namespace = [
            {"type_id": c.type_id, "identifier": c.identifier}
            for c in self.get_children(deep=True)
        ]
        return super().pre_create(db)

    def cleanup(self, db: "Datalayer"):
        """Cleanup hook.

        :param db: Datalayer instance
        """
        super().cleanup(db=db)
        if self.namespace is not None:
            for type_id, identifier in self.namespace:
                db.remove(type_id=type_id, identifier=identifier, force=True)
        return super().cleanup(db)

    @classmethod
    def build_from_db(cls, identifier, db: "Datalayer"):
        """Build application from `superduper`.

        :param identifier: Identifier of the application.
        :param db: Datalayer instance
        """
        components = []
        for component_info in db.show():
            logging.info(f"Component info: {component_info}")
            if not all(
                [
                    isinstance(component_info, dict),
                    "type_id" in component_info,
                    "identifier" in component_info,
                ]
            ):
                raise ValueError("Invalid component info.")

            component = db.load(
                type_id=component_info["type_id"],
                identifier=component_info["identifier"],
            )

            components.append(component)

        # Do not need to include outputs and schema components
        model_outputs_components = set()
        for component in components:
            if any(
                [
                    component.type_id == "table"
                    and component.identifier.startswith(CFG.output_prefix),
                    component.type_id == "schema"
                    and component.identifier.startswith("_schema/"),
                ]
            ):
                logging.info(f"Delete the outputs of {component.identifier}")
                model_outputs_components.add((component.type_id, component.identifier))
                model_outputs_components.update(
                    [(c.type_id, c.identifier) for c in component.children]
                )

        # Do not need to include components with parent
        components_with_parent = set()
        for component in components:
            if component.children:
                logging.info("\n" + "-" * 80)
                logging.info(f"Delete the children of {component.identifier}:")
                logging.info(f"Children: {[c.identifier for c in component.children]}")
                components_with_parent.update(
                    [(c.type_id, c.identifier) for c in component.children]
                )

        remove_components = model_outputs_components | components_with_parent

        logging.info(f"Remove components: {remove_components}")

        components = [
            c for c in components if (c.type_id, c.identifier) not in remove_components
        ]

        if not components:
            raise ValueError("No components found.")

        logging.info("Combine components to application.")
        components_strings = "\n".join(
            [f"{c.type_id}.{c.identifier}" for c in components]
        )
        logging.info(f"Components: \n{components_strings}")
        app = cls(identifier=identifier, components=components)
        app.init(db)
        return app
