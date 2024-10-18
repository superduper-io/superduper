import pprint
import sys

from superduper import CFG, logging
from superduper.base.cursor import SuperDuperCursor


def _prompt(data_backend: str | None = None):
    from superduper import superduper

    data_backend = data_backend or CFG.data_backend

    db = superduper(data_backend)
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
            try:
                exec(input_, values)
            except Exception as e:
                logging.error(str(e))
                continue
            except KeyboardInterrupt:
                print('Aborted')
                sys.exit(0)
            continue

        if '.predict' in input_:
            parts = input_.strip().split('.')
            model = parts[0]
            rest = '.'.join(parts[1:])
            values['model'] = db.load('model', model)
            try:
                exec(f'result = model.{rest}', values)
            except Exception as e:
                logging.error(str(e))
                continue

            pprint.pprint(values['result'])
            continue

        parts = input_.strip().split('.')
        table = parts[0]
        rest = '.'.join(parts[1:])
        try:
            exec(f'result = db["{table}"].{rest}.execute()', values)
        except Exception as e:
            logging.error(str(e))
            continue
        if isinstance(values['result'], SuperDuperCursor):
            for r in values['result']:
                pprint.pprint(r.unpack())
        else:
            pprint.pprint(values['result'].unpack())
