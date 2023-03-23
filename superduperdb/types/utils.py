
def convert_from_bytes_to_types(r, converters=None):
    if converters is None:
        converters = {}  # pragma: no cover
    for k in r:
        if isinstance(r[k], dict):
            if k == '_content':
                converter = converters[r[k]['type']]
                return converter.decode(r[k]['bytes'])
            else:
                r[k] = convert_from_bytes_to_types(r[k], converters=converters)
    return r