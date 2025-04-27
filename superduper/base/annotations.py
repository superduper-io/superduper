import inspect
import typing as t


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

        def decorated(self, *, context: str = '', job: bool = False, **kwargs):
            # the futures kwargs is a placeholder to allow for pinning
            # dependencies according to the readiness of upstream jobs
            if event_types != ('apply',):
                msg = 'Method must be a `Trigger` instance to take non-applying events'
                from superduper.components.cdc import CDC

                assert isinstance(self, CDC), msg
            from superduper.base.metadata import Job

            if job:
                return Job(
                    component=self.__class__.__name__,
                    identifier=self.identifier,
                    uuid=self.uuid,
                    method=f.__name__,
                    kwargs=kwargs,
                    context=context,
                )
            else:
                self.setup()
                signature_params = inspect.signature(f).parameters
                kwargs = {k: v for k, v in kwargs.items() if k in signature_params}
                return f(self, **kwargs)

        decorated.events = event_types
        decorated.depends = depends
        decorated.requires = requires
        return decorated

    return decorator
