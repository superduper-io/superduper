
def convert_from_bytes_to_types(r, converters=None):
    if converters is None:
        converters = {}  # pragma: no cover
    if isinstance(r, dict) and '_content' in r:
        converter = converters[r['_content']['type']]
        return converter.decode(r['_content']['bytes'])
    elif isinstance(r, list):
        return [convert_from_bytes_to_types(x, converters=converters) for x in r]
    elif isinstance(r, dict):
        for k in r:
            r[k] = convert_from_bytes_to_types(r[k], converters=converters)
    return r