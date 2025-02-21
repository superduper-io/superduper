import os
import tempfile

from superduper.base.artifacts import ArtifactStore


def test_bytes(artifact_store: ArtifactStore):
    # test put_bytes
    artifact_store.put_bytes(b"hello", "id")

    # test get_bytes
    assert artifact_store.get_bytes("id") == b"hello"

    # test put same file again
    artifact_store.put_bytes(b"hello_new", "id")
    assert artifact_store.get_bytes("id") == b"hello_new"


def test_file(artifact_store: ArtifactStore):
    _test_file(artifact_store)
    _test_directory(artifact_store)


def _build_directory(path):
    tmpdir_path = os.path.join(path, "test_data")
    os.makedirs(tmpdir_path, exist_ok=True)
    for i in range(10):
        file_name = f"{i}.txt"
        file_path = os.path.join(tmpdir_path, file_name)

        with open(file_path, "w") as file:
            file.write(str(i))

        for j in range(10):
            sub_dir = os.path.join(tmpdir_path, f"subdir_{j}")
            os.makedirs(sub_dir, exist_ok=True)
            sub_file_path = os.path.join(sub_dir, file_name)
            with open(sub_file_path, "w") as file:
                file.write(f"{i} {j}")

    return tmpdir_path


def _create_file(path):
    file_path = os.path.join(path, "test_data.txt")
    with open(file_path, "w") as file:
        file.write("hello")
    return file_path


def _test_file(artifact_store: ArtifactStore):
    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_file(tempdir)
        new_file_id = artifact_store.put_file(path, "id-file")
        loaded_file_path = artifact_store.get_file(new_file_id)
        with open(loaded_file_path, "r") as file:
            assert file.read() == "hello"

    artifact_store._delete_bytes(new_file_id)

    import pytest

    with pytest.raises(FileNotFoundError):
        loaded_file_path = artifact_store.get_file(new_file_id)


def _test_directory(artifact_store: ArtifactStore):
    import filecmp

    def compare_directories(path1, path2):
        comparison = filecmp.dircmp(path1, path2)
        assert not comparison.diff_files
        for sub_dir in comparison.subdirs:
            compare_directories(
                os.path.join(path1, sub_dir), os.path.join(path, sub_dir)
            )

        for file in comparison.common_files:
            with open(os.path.join(path1, file), "r") as file1, open(
                os.path.join(path2, file), "r"
            ) as file2:
                if file1.read() != file2.read():
                    return False
        return True

    with tempfile.TemporaryDirectory() as tempdir:
        path = _build_directory(tempdir)
        new_file_id = artifact_store.put_file(path, "id-dir")
        loaded_file_path = artifact_store.get_file(new_file_id)

        assert compare_directories(path, loaded_file_path)

    artifact_store._delete_bytes(new_file_id)

    import pytest

    with pytest.raises(FileNotFoundError):
        loaded_file_path = artifact_store.get_file(new_file_id)
