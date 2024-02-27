import hashlib
import os


def hash_string(string: str):
    return hashlib.sha256(string.encode()).hexdigest()


def hash_dict(data: dict):
    def process(d):
        if isinstance(d, dict):
            return sorted((k, process(v)) for k, v in d.items())
        elif isinstance(d, set):
            return sorted(d)
        else:
            return d

    json_string = str(process(data))
    return hash_string(json_string)


def random_sha1():
    """
    Generate random sha1 values
    Can be used to generate file_id and other values
    """
    random_data = os.urandom(256)
    sha1 = hashlib.sha1()
    sha1.update(random_data)
    return sha1.hexdigest()
