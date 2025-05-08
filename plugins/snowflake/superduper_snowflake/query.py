from snowflake.snowpark.functions import col
from superduper import CFG
from superduper.base.query import Op, Query


def map_superduper_query_to_snowpark_query(session, query, primary_id: str = 'id'):
    """Map a SuperDuper query to a Snowpark query.

    :param session: The Snowpark session.
    :param query: The SuperDuper query.
    :param primary_id: The primary ID column.
    """
    q = session.table(f'"{query.table}"')

    if not query.parts:
        return q

    for i, part in enumerate(query.parts):

        if isinstance(part, str):
            assert i == 0
            if part == 'primary_id':
                q = col(f'"{primary_id}"')
            else:
                q = col(f'"{part}"')
            continue

        if isinstance(part, Op):
            q = getattr(q, part.name)(part.args[0])
            continue

        if part.name == 'select':
            args = list(part.args[:])
            for i, a in enumerate(args):
                if isinstance(a, Query):
                    assert str(a) == f'{query.table}.primary_id'
                    args[i] = primary_id
            args = [f'"{a}"' for a in args]

            if args:
                predict_ids = (
                    query.decomposition.outputs.args
                    if query.decomposition.outputs
                    else []
                )
                args.extend(
                    [
                        f'"{CFG.output_prefix}{pid}"'
                        for pid in predict_ids
                        if f'"{CFG.output_prefix}{pid}"' not in args
                    ]
                )
                q = q.select(*args)

            continue

        if part.name == 'filter':
            args = list(part.args[:])
            kwargs = part.kwargs.copy()

            for i, a in enumerate(args):
                if isinstance(a, Query):
                    args[i] = map_superduper_query_to_snowpark_query(
                        session, a, primary_id
                    )

            for k, v in kwargs.items():
                if isinstance(v, Query):
                    kwargs[k] = map_superduper_query_to_snowpark_query(
                        session, v, primary_id
                    )

            condition = args[0]
            for next_condition in args[1:]:
                condition = condition & next_condition
            q = getattr(q, part.name)(condition, **kwargs)

            continue

        if part.name == 'outputs':
            output_tables = []
            for arg in part.args:
                t = session.table(f'"{CFG.output_prefix + arg}"')
                cols = [c for c in t.columns if c != '"id"']
                t = t.select(*cols)
                t = t.with_column_renamed('"_source"', f'"_source_{arg}"')
                output_tables.append((arg, t))

            for arg, table in output_tables:
                q = q.join(
                    table,
                    q[f'"{primary_id}"'] == table[f'"_source_{arg}"'],
                    join_type="left",
                )
            continue

        q = getattr(q, part.name)(*part.args, **part.kwargs)

    return q
