import io
import pickle


def load(file_id, filesystem):
    bytes_ = filesystem.get(file_id).read()
    f = io.BytesIO(bytes_)
    return pickle.load(f)


def save(object, filesystem):
    with io.BytesIO() as f:
        pickle.dump(object, f)
        bytes_ = f.getvalue()
    return filesystem.put(bytes_)
