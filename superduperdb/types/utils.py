
def convert_types(r, converters=None):
    if converters is None:
        converters = {}  # pragma: no cover
    for k in r:
        if isinstance(r[k], dict):
            if '_content' in r[k]:
                converter = converters[r[k]['_content']['converter']]
                r[k] = converter.decode(r[k]['_content']['bytes'])
            else:
                convert_types(r[k], converters=converters)
    return r