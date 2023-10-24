from superduperdb.backends.mongodb import query as q
from superduperdb.component.document import Document


def test_select_missing_outputs(random_data):
    docs = list(random_data.execute(q.Collection('documents').find({}, {'_id': 1})))
    ids = [r['_id'] for r in docs[: len(docs) // 2]]
    random_data.execute(
        q.Collection('documents').update_many(
            {'_id': {'$in': ids}},
            Document({'$set': {'_outputs.x.test_model_output': 'test'}}),
        )
    )

    select = q.Collection('documents').find({}, {'_id': 1})
    modified_select = select.select_ids_of_missing_outputs('x', 'test_model_output')

    out = list(random_data.execute(modified_select))
    assert len(out) == (len(docs) - len(ids))
