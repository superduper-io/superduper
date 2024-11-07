import json

import pytest

from superduper import Application, Document


@pytest.mark.skip
def test_build_from_template(db):
    from superduper import templates

    db.apply(templates.simple_rag)

    with open('test/material/sample_app/component.json') as f:
        component = json.load(f)

    component = templates.simple_rag.form_template
    component['_variables']['output_prefix'] = '_output__'

    c = Document.decode(component, db=db).unpack()

    assert isinstance(c, Application)
