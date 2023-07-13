from sddb.models import loading


converters = {}


def encode(handle, bytes_):
    if handle not in converters:
        converters[handle] = loading.load({'path': handle})
    return converters[handle].encode(bytes_)


def decode(handle, bytes_):
    if handle not in converters:
        converters[handle] = loading.load({'path': handle})
    return converters[handle].decode(bytes_)
