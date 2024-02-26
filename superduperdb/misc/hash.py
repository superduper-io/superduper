import hashlib


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
