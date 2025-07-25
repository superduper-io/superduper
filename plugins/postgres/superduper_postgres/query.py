from superduper.base.query import Query
from superduper import logging
from superduper import CFG


OP_LOOKUP = {
    '==': '=',
    '!=': '<>',
    '<': '<',
    '<=': '<=',
    '>': '>',
    '>=': '>=',
    'in': 'IN',
}


def map_superduper_query_to_postgres_query(query: Query):
    logging.info(f'Mapping SuperDuper query {query} to Postgres query')

    assert query.type == 'select'

    d = query.decomposition

    cols = list(d.select.args if d.select else query.decomposition.columns)
    for i, c in enumerate(cols):
        if isinstance(c, Query):
            cols[i] = c.execute()
        else:
            cols[i] = f'"{c}"'

    pid = query.primary_id.execute()

    if d.outputs:
        predict_ids = d.outputs.args
        for i, col in enumerate(cols):
            try:
                next(predict_id for predict_id in predict_ids if predict_id in col)
                cols[i] = f'{col}.{col}'
            except StopIteration:
                cols[i] = f'{query.table}.{col}'

    cols = ', '.join(cols)

    output = f'SELECT {cols} FROM "{d.table}"'
    if d.outputs:
        for predict_id in d.outputs.args:
            output_t = f'{CFG.output_prefix}{predict_id}'
            output += f' INNER JOIN {output_t} ON {d.table}.{pid} = {output_t}._source \n'

    filter_str = ''
    if d.filter:
        filters = [f.parts for f in d.filter.args]
        filter_parts = []  
        for col, f in filters:
            if f.symbol in OP_LOOKUP:
                value = f.args[0]
                if isinstance(value, str):
                    value = f"'{value}'"

                if isinstance(value, Query):
                    value = f'"{value.execute()}"'

                if f.symbol == 'in':
                    if col == 'primary_id':
                        col = query.primary_id.execute()
                    assert all(isinstance(v, str) for v in value), "All values in 'in' operator must be strings"
                    value = '(' + ','.join([f"'{v}'" for v in value]) + ')'  # Assuming value is a list for 'in' operator

                filter_parts.append(
                    f"\"{col}\" {OP_LOOKUP[f.symbol]} {value}"
                )

        filter_str = ' WHERE ' + ' AND '.join(filter_parts)

    output += filter_str

    if d.limit:
        output += f' LIMIT {d.limit.args[0]}'

    if d.limit and 'offset' in d.limit.kwargs:
        output += f' OFFSET {d.limit.kwargs["offset"]}'

    logging.info(f'Mapped SuperDuper to Postgres query: {output}')

    return output