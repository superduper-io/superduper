def convert_from_bytes_to_types(r, types):
    """
    Convert the bson byte objects in a nested dictionary into python objects.

    :param r: dictionary potentially containing non-Bsonable content
    """
    if isinstance(r, dict) and '_content' in r:
        converter = types[r['_content']['type']]
        return converter.decode(r['_content']['bytes'])
    elif isinstance(r, list):
        return [convert_from_bytes_to_types(x, types) for x in r]
    elif isinstance(r, dict):
        for k in r:
            r[k] = convert_from_bytes_to_types(r[k], types)
    return r


def convert_from_types_to_bytes(r, types, type_lookup):
    """
    Convert the non-Bsonable python objects in a nested dictionary into ``bytes``

    :param r: dictionary potentially containing non-Bsonable content
    """
    if isinstance(r, dict):
        for k in r:
            r[k] = convert_from_types_to_bytes(r[k], types, type_lookup)
        return r
    try:
        t = type_lookup[type(r)]
    except KeyError:
        t = None
    if t is not None:
        return {'_content': {'bytes': types[t].encode(r), 'type': t}}
    return r
