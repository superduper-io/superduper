import pprint
import sys

from superduper.base.cursor import SuperDuperCursor


def _prompt():
    from superduper import superduper

    db = superduper()
    values = {'db': db}

    while True:
        input_ = input('superduper> ')

        if input_.strip() == '/s':
            import pandas

            t = db.show()
            print(pandas.DataFrame(t))
            continue

        if input_.strip() == '/q':
            print('Aborted')
            sys.exit(0)

        if input_.startswith('/'):
            import pandas

            out = db.show(input_.strip()[1:])
            print(pandas.Series(out).to_string())
            continue

        import re

        if re.match('[A-Za-z0-9_]+ = ', input_):
            exec(input_, values)
            continue

        if '.predict' in input_:
            parts = input_.strip().split('.')
            model = parts[0]
            rest = '.'.join(parts[1:])
            values['model'] = db.load('model', model)
            exec(f'result = model.{rest}', values)
            pprint.pprint(values['result'])
            continue

        parts = input_.strip().split('.')
        table = parts[0]
        rest = '.'.join(parts[1:])
        exec(f'result = db["{table}"].{rest}.execute()', values)
        if isinstance(values['result'], SuperDuperCursor):
            for r in values['result']:
                pprint.pprint(r.unpack())
        else:
            pprint.pprint(values['result'].unpack())
