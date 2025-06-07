import typing as t
from contextlib import contextmanager

from superduper import CFG, logging

from .component import Component, build_vars_var

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


# 3. The context-manager that temporarily sets the variable
@contextmanager
def build_context(vars_dict: dict[str, t.Any] | None):
    """Context manager to set build variables for components.

    :param vars_dict: Dictionary of variables to set for the build context.
    """
    token = build_vars_var.set(vars_dict or {})
    try:
        yield
    finally:
        build_vars_var.reset(token)


class Application(Component):
    """
    A placeholder to hold list of components with associated funcionality.

    Components are sorted in a way that respects their mutual dependencies.

    :param components: List of components to group together and apply to `superduper`.
    :param build_variables: Variables which were supplied to a template to build.
    :param build_template: Template which was used to build.
    :param variables: Variables which are used inside the application.
    """

    breaks: t.ClassVar[t.Sequence[str]] = ('components',)
    component_cache: t.ClassVar[bool] = True

    components: t.List[Component]
    build_variables: t.Dict | None = None
    build_template: str | None = None
    variables: t.Dict | None = None

    def postinit(self):
        """Post-initialization method to set up the application."""
        with build_context(self.variables):
            for component in self.components:
                # Might be just a ComponentRef
                if isinstance(component, Component):
                    component.postinit()
            return super().postinit()

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
                    "component" in component_info,
                    "identifier" in component_info,
                ]
            ):
                raise ValueError("Invalid component info.")

            component = db.load(
                component=component_info["component"],
                identifier=component_info["identifier"],
            )
            components.append(component)

        # Do not need to include outputs and schema components
        model_outputs_components = set()
        for component in components:
            if any(
                [
                    component.component == "Table"
                    and component.identifier.startswith(CFG.output_prefix),
                ]
            ):
                logging.info(f"Delete the outputs of {component.identifier}")
                model_outputs_components.add(
                    (component.component, component.identifier)
                )
                model_outputs_components.update(
                    [(c.component, c.identifier) for c in component.children]
                )

        # Do not need to include components with parent
        components_with_parent = set()
        for component in components:
            if component.children:
                logging.info("\n" + "-" * 80)
                logging.info(f"Delete the children of {component.identifier}:")
                logging.info(f"Children: {[c.identifier for c in component.children]}")
                components_with_parent.update(
                    [(c.component, c.identifier) for c in component.children]
                )

        remove_components = model_outputs_components | components_with_parent

        logging.info(f"Remove components: {remove_components}")

        components = [
            c
            for c in components
            if (c.component, c.identifier) not in remove_components
        ]

        if not components:
            raise ValueError("No components found.")

        logging.info("Combine components to application.")
        components_strings = "\n".join(
            [f"{c.component}.{c.identifier}" for c in components]
        )
        logging.info(f"Components: \n{components_strings}")
        app = cls(identifier=identifier, components=components)
        app.setup(db)
        return app
