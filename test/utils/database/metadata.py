import uuid

import pytest

from superduper.base.event import Job
from superduper.base.metadata import MetaDataStore


def test_component(metadata: MetaDataStore):
    _test_show_components(metadata)
    _test_get_component(metadata)
    _test_delete_component_version(metadata)


def test_parent_child(metadata: MetaDataStore):
    parent_component = {
        "identifier": "parent-1",
        "type_id": "parent",
        "version": 1,
        "_path": "superduper.container.model.Model",
        "uuid": str(uuid.uuid4()),
    }

    child_component = {
        "identifier": "child-1",
        "type_id": "child",
        "version": 1,
        "_path": "superduper.container.model.Model",
        "uuid": str(uuid.uuid4()),
    }

    not_child_component = {
        "identifier": "not-child-1",
        "type_id": "not-child",
        "version": 1,
        "_path": "superduper.container.model.Model",
        "uuid": str(uuid.uuid4()),
    }

    metadata.create_component(parent_component)
    metadata.create_component(child_component)
    metadata.create_component(not_child_component)

    metadata.create_parent_child(
        parent_component["uuid"],
        child_component["uuid"],
    )

    # test component_version_has_parents
    assert metadata.component_version_has_parents("child", "child-1", 1)
    assert not metadata.component_version_has_parents("not-child", "not-child-1", 1)

    # test get_component_version_parents
    parents = metadata.get_component_version_parents(child_component["uuid"])
    assert parents == [parent_component["uuid"]]

    # test delete_parent_child
    metadata.delete_parent_child(parent_component["uuid"], child_component["uuid"])

    assert not metadata.component_version_has_parents("child", "child-1", 1)


def test_job(metadata: MetaDataStore):
    for i in range(5):
        job = Job(
            identifier=str(i),
            job_id=str(i),
            args=(1, 2),
            kwargs={"message": str(i)},
            uuid='myuuid',
            type_id='listener',
            method='run',
            context='mycontenxt',
        )
        metadata.create_job(job.dict())

    # test show_jobs
    jobs = metadata.show_jobs()
    assert len(jobs) == 5

    # test get_job
    job_get = metadata.get_job("3")
    assert job_get is not None
    assert job_get["identifier"] == "3"
    assert job_get["status"] == "pending"
    assert job_get["args"] == [1, 2]
    assert job_get["kwargs"] == {"message": "3"}

    job_get = metadata.get_job("100")
    assert job_get is None

    # test update_job
    metadata.update_job("3", "status", "running")
    job_get = metadata.get_job("3")
    assert job_get["status"] == "running"


def test_artifact_relation(metadata: MetaDataStore):
    uuid_1 = str(uuid.uuid4())
    artifact_ids_1 = ["artifact-1", "artifact-2"]
    uuid_2 = str(uuid.uuid4())
    artifact_ids_2 = ["artifact-3", "artifact-4"]

    metadata.create_artifact_relation(uuid_1, artifact_ids_1)
    metadata.create_artifact_relation(uuid_2, artifact_ids_2)

    assert metadata.get_artifact_relations(uuid=uuid_1) == artifact_ids_1
    assert metadata.get_artifact_relations(artifact_id="artifact-1") == [uuid_1]
    assert metadata.get_artifact_relations(artifact_id="artifact-2") == [uuid_1]

    assert metadata.get_artifact_relations(uuid=uuid_2) == artifact_ids_2
    assert metadata.get_artifact_relations(artifact_id="artifact-3") == [uuid_2]
    assert metadata.get_artifact_relations(artifact_id="artifact-4") == [uuid_2]

    metadata.delete_artifact_relation(uuid_1, artifact_ids_1[0])
    assert metadata.get_artifact_relations(uuid=uuid_1) == [artifact_ids_1[1]]

    assert metadata.get_artifact_relations(uuid=uuid_2) == artifact_ids_2


def _create_components(type_ids, identifiers, versions, metadata):
    versions = versions or [0]
    uuid2component = {}
    for type_id in type_ids:
        for identifier in identifiers:
            for version in versions:
                uuid_value = str(uuid.uuid4())
                component = {
                    "identifier": identifier,
                    "type_id": type_id,
                    "version": version,
                    "_path": "superduper.container.model.Model",
                    "uuid": uuid_value,
                }
                uuid2component[uuid_value] = component
                metadata.create_component(component.copy())

    return uuid2component


def _test_show_components(metadata: MetaDataStore):
    _create_components(["show-1", "show-2"], ["show-a", "show-b"], [0, 1], metadata)
    _create_components(["show-3"], ["show-c", "show-d"], [0, 1], metadata)

    assert metadata.show_components("show-1") == ["show-a", "show-b"]
    assert metadata.show_components("show-2") == ["show-a", "show-b"]
    assert metadata.show_components("show-3") == ["show-c", "show-d"]
    show_components = metadata.show_components()
    assert show_components is not None
    assert len(show_components) == 6
    assert metadata.show_components() == [
        {"type_id": "show-1", "identifier": "show-a"},
        {"type_id": "show-1", "identifier": "show-b"},
        {"type_id": "show-2", "identifier": "show-a"},
        {"type_id": "show-2", "identifier": "show-b"},
        {"type_id": "show-3", "identifier": "show-c"},
        {"type_id": "show-3", "identifier": "show-d"},
    ]


def _test_get_component(metadata: MetaDataStore):
    uuid2component = _create_components(
        ["get-1", "get-2"], ["get-a", "get-b"], [0, 1], metadata
    )

    # test default version
    r = metadata.get_component("get-1", "get-a")
    assert r["type_id"] == "get-1"
    assert r["identifier"] == "get-a"
    assert r["version"] == 1

    # test specific version
    r = metadata.get_component("get-1", "get-a", 0)
    assert r["type_id"] == "get-1"
    assert r["identifier"] == "get-a"
    assert r["version"] == 0

    for uuid_value, component_expected in uuid2component.items():
        component = metadata.get_component_by_uuid(uuid_value)
        for key, value in component_expected.items():
            assert component[key] == value

    _create_components(["get-3"], ["get-c"], [0, 1, 4, 10], metadata)
    # test_last_version
    r = metadata.get_component("get-3", "get-c")
    assert r["version"] == 10

    # show_component_versions
    versions = metadata.show_component_versions("get-3", "get-c")
    assert versions == [0, 1, 4, 10]


def _test_delete_component_version(metadata: MetaDataStore):
    _create_components(
        ["delete-1", "delete-2"], ["delete-a", "delete-b"], [0, 1], metadata
    )

    metadata.delete_component_version("delete-1", "delete-a", 0)
    metadata.delete_component_version("delete-1", "delete-a", 1)
    assert metadata.show_components("delete-1") == ["delete-b"]
    assert metadata.show_components("delete-2") == ["delete-a", "delete-b"]

    with pytest.raises(FileNotFoundError):
        metadata.get_component("delete-1", "delete-a")

    metadata.delete_component_version("delete-2", "delete-a", 1)
    component = metadata.get_component("delete-2", "delete-a")
    assert component["version"] == 0

    assert metadata.show_components("delete-2") == ["delete-a", "delete-b"]
