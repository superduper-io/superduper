import dataclasses as dc
import typing as t

from superduperdb.base.document import Document
from superduperdb.components.datatype import pickle_serializer
from superduperdb.misc.annotations import merge_docstrings

from .component import Component, ensure_initialized

if t.TYPE_CHECKING:
    from superduperdb import DataType


@merge_docstrings
@dc.dataclass(kw_only=True)
class Template(Component):
    """
    Application template component.

    :param template: Template.
    :param info: Info.
    """

    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('component', pickle_serializer),
    )
    type_id: t.ClassVar[str] = 'template'

    component: t.Union[Component, t.Dict]
    info: t.Dict

    def __post_init__(self, db, artifacts):
        if isinstance(self.component, Component):
            self.component = self.component.encode()
        return super().__post_init__(db, artifacts)

    def pre_create(self, db):
        """
        Database `pre_create` hook.

        :param db: Datalayer instance.
        """
        assert isinstance(self.component, dict)
        from superduperdb.misc.special_dicts import SuperDuperFlatEncode

        if not isinstance(self.component, SuperDuperFlatEncode):
            self.component = SuperDuperFlatEncode(self.component)
        if self.component.blobs:
            for file_id, blob in self.component.blobs.items():
                self.db.artifact_store.put_bytes(blob, file_id=file_id)
            self.component.pop_blobs()
        for file_id, file_path in self.component.files:
            self.db.artifact_store.put_file(file_path=file_path, file_id=file_id)

    @ensure_initialized
    def __call__(self, **kwargs):
        """Method to create component from the given template and `kwargs`."""
        assert set(kwargs.keys()) == (set(self.info.keys())), 'Invalid variables'
        t = Document.decode(self.component, db=self.db).unpack()
        t.init(db=self.db)
        return t.set_variables(db=self.db, **kwargs)
