import os

from superduper import CFG, superduper, templates

skips = []


def test_template():
    CFG.auto_schema = True

    db = superduper()

    template_name = os.environ['SUPERDUPER_TEMPLATE']

    if template_name in skips:
        print(f'Skipping template {template_name}')
        return

    t = getattr(templates, template_name)

    db.apply(t)

    assert f'sample_{template_name}' in db.show('table')

    sample = db[f'sample_{template_name}'].select().limit(2).tolist()

    assert sample

    print('Got sample:', sample)
    print(f'Got {len(sample)} samples')

    app = t()

    db.apply(app)
