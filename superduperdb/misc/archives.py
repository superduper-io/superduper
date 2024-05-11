import os
import tarfile


# TODO: Remove the unused functions
def to_tarball(folder_path: str, output_path: str):
    """Create a tarball (compressed archive) from a folder.

    :param folder_path: Path to the folder to be archived.
    :param output_path: Path to the output tarball file.
    """
    try:
        with tarfile.open(output_path + '.tar.gz', "w:gz") as tar:
            for item in os.listdir(folder_path):
                full_path = os.path.join(folder_path, item)
                tar.add(full_path, arcname=item)
    except Exception as e:
        print(f"An error occurred: {e}")


def from_tarball(tarball_path: str):
    """Extract the contents of stack tarball.

    :param tarball_path: Path to the tarball file.
    """
    extract_path = tarball_path.split('.tar.gz')[0]
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    return extract_path
