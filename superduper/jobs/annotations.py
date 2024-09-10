import inspect
import typing as t

from superduper.jobs.job import Job


def trigger(
    *event_types: t.Sequence[str],
    depends: t.Sequence[str] | str = (),
    requires: t.Sequence[str] | str = (),
):
    """Decorator to trigger a method when an event is detected.

    :param event_types: Event to trigger the method.
    :param depends: Triggers which should run before this method.
    :param requires: Dataclass parameters/ attributes which should be
                     available to trigger the method
    """
    if isinstance(depends, str):
        depends = [depends]

    if isinstance(requires, str):
        requires = [requires]

    def decorator(f):
        """Decorator to trigger a method when an event of type."""
        takes_ids = 'ids' in inspect.signature(f).parameters
        if event_types != ('apply',):
            assert takes_ids, (
                f"Method {f.__name__} must take an 'ids' argument"
                " to be triggered by anything apart from 'apply'"
            )

        def decorated(self, ids: t.List[str] | None = None, job: bool = False):
            if job:
                return Job(
                    type_id=self.type_id,
                    identifier=self.identifier,
                    uuid=self.uuid,
                    method=f.__name__,
                    args=(ids,) if takes_ids else (),
                    db=self.db,
                )
            else:
                return f(self, ids) if takes_ids else f(self)

        decorated.events = event_types
        decorated.depends = depends
        decorated.requires = requires
        return decorated

    return decorator
