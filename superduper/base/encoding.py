import typing as t
from dataclasses import dataclass, field

from superduper.base.config import Config

if t.TYPE_CHECKING:
    from superduper.base.base import Base
    from superduper.base.datalayer import Datalayer


@dataclass
class EncodeContext:
    """
    Context for encoding and decoding data.

    :param name: Name of the context.
    :param builds: A dictionary of builds.
    :param blobs: A dictionary of blobs.
    :param files: A dictionary of files.
    :param db: A Datalayer instance.
    :param leaves_to_keep: A sequence of Base instances to keep.
    :param metadata: Whether to include metadata.
    :param defaults: Whether to include defaults.
    :param cfg: Configuration object.
    :param keep_variables: Whether to keep variables.
    :param include_defaults: Whether to include default values.
    """

    name: str = '__main__'
    builds: t.Dict[str, dict] = field(default_factory=dict)
    blobs: t.Dict[str, bytes] = field(default_factory=dict)
    files: t.Dict[str, str] = field(default_factory=dict)
    db: t.Optional['Datalayer'] = None
    leaves_to_keep: t.Sequence['Base'] = field(default_factory=tuple)
    metadata: bool = True
    defaults: bool = True
    cfg: t.Optional[Config] = None
    keep_variables: bool = False
    include_defaults: bool = True

    def __call__(self, name: str):
        return EncodeContext(
            name=name,
            builds=self.builds,
            blobs=self.blobs,
            files=self.files,
            db=self.db,
            leaves_to_keep=self.leaves_to_keep,
            metadata=self.metadata,
            defaults=self.defaults,
            keep_variables=self.keep_variables,
            include_defaults=self.include_defaults,
            cfg=self.cfg,
        )
