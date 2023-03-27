
def load(file_id, filesystem):
    return filesystem.get(file_id).read()


def save(bytes_, filesystem):
    return filesystem.put(bytes_)

