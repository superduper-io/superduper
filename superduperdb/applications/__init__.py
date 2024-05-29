import typing as t

from superduperdb.base.datalayer import Datalayer


def __getattr__(name):
    from superduperdb import Application

    ds = f"""
        Build {name} application.

        :param kwargs: key-values pairs passed to `Variable` instances in 
            template.
        """

    def function(identifier: str, db: t.Optional[Datalayer] = None, **kwargs):
        return Application(  # type: ignore[call-arg]
            identifier=identifier,
            template=name,
            kwargs=kwargs,
            db=db,
        )

    function.__name__ = name
    function.__doc__ = ds
    return function
