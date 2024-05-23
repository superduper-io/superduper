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
