import dataclasses as dc
import typing as t

from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.components.datatype import dict_serializer, pickle_serializer
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
        ('template', pickle_serializer),
    )
    type_id: t.ClassVar[str] = 'template'

    template: t.Union[Component, t.Dict]
    info: t.Dict

    def __post_init__(self, db, artifacts):
        if isinstance(self.template, Component):
            self.template = self.template.encode()
        return super().__post_init__(db, artifacts)

    def pre_create(self, db):
        if self.template.blobs:
            for file_id, blob in self.template.blobs.items():
                self.db.artifact_store.put_bytes(blob, file_id=file_id)
            self.template.pop_blobs()
        for file_id, file_path in self.template.files:
            self.db.artifact_store.put_file(file_path=file_path, file_id=file_id)

    @ensure_initialized
    def __call__(self, **kwargs):
        assert set(kwargs.keys()) == (set(self.info.keys())), 'Invalid variables'
        t = Document.decode(self.template, db=self.db).unpack()
        t.init()
        return t.set_variables(db=self.db, **kwargs)