import dataclasses as dc
import os
import typing as t
from test.db_config import DBConfig

import pytest

from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.components.component import Component
from superduperdb.components.datatype import (
    DataType,
    file_serializer,
    serializers,
)


@dc.dataclass(kw_only=True)
class TestComponent(Component):
    path: str
    type_id: t.ClassVar[str] = "TestComponent"

    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, "DataType"]]] = (
        ("path", file_serializer),
    )


@pytest.fixture
def artifact_strore(tmpdir) -> FileSystemArtifactStore:
    artifact_strore = FileSystemArtifactStore(f"{tmpdir}")
    artifact_strore._serializers = serializers
    return artifact_strore


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_save_and_load_file(db, artifact_strore: FileSystemArtifactStore):
    db.artifact_store = artifact_strore
    test_component = TestComponent(path="superduperdb", identifier="test")
    db.add(test_component)
    test_component_loaded = db.load("TestComponent", "test")
    assert test_component.path != test_component_loaded.path
    assert os.path.getsize(test_component.path) == os.path.getsize(
        test_component_loaded.path
    )
