import re


def _parse_query_part(part, documents, artifacts, query):
    from superduperdb.backends.mongodb.query import Collection

    part = part.replace(' ', '').replace('\n', '')
    part = part.split('.')
    for i, comp in enumerate(part):
        if i == 0:
            current = Collection(comp)
        else:
            match = re.match('^([a-zA-Z0-9_]+)\((.*)\)$', comp)
            if match is None:
                current = getattr(current, comp)
                continue
            if not match.groups()[1].strip():
                current = getattr(current, match.groups()[0])()
                continue

            comp = getattr(current, match.groups()[0])
            args_kwargs = [x.strip() for x in match.groups()[1].split(',')]
            args = []
            kwargs = {}
            for x in args_kwargs:
                if '=' in x:
                    k, v = x.split('=')
                    # if v.startswith('$documents'):
                    #     v = _get_item(v, documents)
                    # elif v.startswith('$artifacts'):
                    #     v = _get_item(v, artifacts)
                    kwargs[k] = eval(v)
                else:
                    args.append(eval(x))
            current = comp(*args, **kwargs)
    return current


def parse_query(query, documents, artifacts):
    for i, q in enumerate(query):
        query[i] = _parse_query_part(q, documents, artifacts, query[:i])
    return query[-1]


if __name__ == '__main__':
    q = parse_query(
        [
            'documents.find($documents[0], a={"b": 1}).sort(c=1).limit(1)',
        ],
        [],
        [],
    )

    print(q)
