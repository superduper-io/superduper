import os

from superduper import CFG, Template, superduper

skips = []


def test_template():
    CFG.auto_schema = True

    db = superduper()

    template_name = os.environ['SUPERDUPER_TEMPLATE']

    if template_name in skips:
        print(f'Skipping template {template_name}')
        return

    t = Template.read(f'templates/{template_name}')

    db.apply(t)

    assert f'sample_{template_name}' in db.show('Table')

    sample = db[f'sample_{template_name}'].limit(2).execute()

    assert sample

    print('Got sample:', sample)

    app = t()

    db.apply(app)
