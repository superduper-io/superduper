import inspect
import typing as t

from superduper.base.event import EventType
from superduper.jobs.job import ComponentJob


def trigger(*events: t.Sequence[EventType], depends: t.Sequence[str] | str = (), requires: t.Sequence[str] | str = ()):
    """Decorator to trigger a method when an event of type 
    {'update', 'insert', 'delete', 'apply'} are called.

    :param events: Event to trigger the method.
    :param depends: Triggers which should run before this method.
    :param requires: Dataclass parameters/ attributes which should be 
                     available to trigger the method
    """

    if isinstance(depends, str):
        depends = [depends]

    if isinstance(requires, str):
        requires = [requires]

    def decorator(f):
        """Decorator to trigger a method when an event of type.
        """
        takes_ids = 'ids' in inspect.signature(f).parameters
        if events != ('apply',):
            assert takes_ids, (
                f"Method {f.__name__} must take an 'ids' argument"
                " to be triggered by anything apart from 'apply'"
            )

        def decorated(self, ids: t.List[str] | None = None, job: bool = False):

            if job:
                return ComponentJob(
                    component_identifier=self.identifier,
                    component_uuid=self.uuid,
                    type_id=self.type_id,
                    method_name=f.__name__,
                    args=(ids,) if takes_ids else (),
                    db=self.db,
                )
            else:
                return f(self, ids) if takes_ids else f(self)

        decorated.events = events
        decorated.depends = depends
        decorated.requires = requires
        return decorated

    return decorator