import hashlib
import os


def hash_string(string: str):
    """Hash a string.

    :param string: string to hash
    """
    return hashlib.sha256(string.encode()).hexdigest()


def random_sha1():
    """Generate random sha1 values."""
    random_data = os.urandom(256)
    sha1 = hashlib.sha1()
    sha1.update(random_data)
    return sha1.hexdigest()


def hash_path(path, hash_type="md5"):
    """Hash a file or folder.

    Only the metadata of the file/folder is hashed, not the content.

    Metadata includes:
    - All file names in the folder
    - All file sizes
    - All file modification times

    :param path: Path to the file or folder.
    :param hash_type: Hashing algorithm to use. Default is md5.
    """
    hash_func = getattr(hashlib, hash_type)()
    hash_func.update(os.path.abspath(path).encode('utf-8'))
    if os.path.isfile(path):
        size = os.path.getsize(path)
        modification_time = os.path.getmtime(path)
        filename = os.path.basename(path)
        metadata_string = f"{filename}{size}{modification_time}"
        hash_func.update(metadata_string.encode('utf-8'))
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            files.sort()
            for filename in files:
                file_path = os.path.join(root, filename)
                size = os.path.getsize(file_path)
                modification_time = os.path.getmtime(file_path)
                metadata_string = f"{filename}{size}{modification_time}"
                hash_func.update(metadata_string.encode('utf-8'))
    else:
        raise ValueError("Provided path does not exist or is not a file/folder.")

    return hash_func.hexdigest()
