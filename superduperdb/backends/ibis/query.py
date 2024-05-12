import dataclasses as dc
import typing as t
import uuid
from collections import defaultdict

import pandas

from superduperdb.backends.base.query import Query, applies_to
from superduperdb.base.cursor import SuperDuperCursor
from superduperdb.components.schema import Schema


def _model_update_impl_flatten(
    db,
    ids: t.List[t.Any],
    predict_id: str,
    outputs: t.Sequence[t.Any],
):
    table_records = []

    for ix in range(len(outputs)):
        for r in outputs[ix]:
            d = {
                '_input_id': str(ids[ix]),
                '_source': str(ids[ix]),
                'output': r,
            }
            table_records.append(d)

    for r in table_records:
        if isinstance(r['output'], dict) and '_content' in r['output']:
            r['output'] = r['output']['_content']['bytes']

    db.databackend.insert(f'_outputs.{predict_id}', table_records)


def _model_update_impl(
    db,
    ids: t.List[t.Any],
    predict_id: str,
    outputs: t.Sequence[t.Any],
    flatten: bool = False,
):
    if not outputs:
        return

    if flatten:
        return _model_update_impl_flatten(
            db, ids=ids, predict_id=predict_id, outputs=outputs
        )

    table_records = []
    for ix in range(len(outputs)):
        d = {
            '_input_id': str(ids[ix]),
            'output': outputs[ix],
        }
        table_records.append(d)

    for r in table_records:
        if isinstance(r['output'], dict) and '_content' in r['output']:
            r['output'] = r['output']['_content']['bytes']

    db.databackend.insert(f'_outputs.{predict_id}', table_records)


@dc.dataclass(kw_only=True, repr=False)
class IbisQuery(Query):
    flavours: t.ClassVar[t.Dict[str, str]] = {
        'pre_like': '^.*\.like\(.*\)\.find',
        'post_like': '^.*\.([a-z]+)\(.*\)\.like(.*)$',
        'insert': '^[^\(]+\.insert\(.*\)$',
    }

    @property
    @applies_to('insert')
    def documents(self):
        return self.parts[0][1][0]

    @property
    def tables(self):
        out = {self.identifier: self.db.tables[self.identifier]}
        for part in self.parts:
            for a in part[1]:
                if isinstance(a, IbisQuery):
                    out.update(a.tables)
            for v in part[2].values():
                if isinstance(v, IbisQuery):
                    out.update(v.tables)
        return out

    @property
    def schema(self):
        fields = {}
        import pdb

        pdb.set_trace()
        t = self.db.load('table', 'documents')
        tables = self.tables
        if len(tables) == 1:
            return self.db.tables[self.identifier].schema
        for t, c in self.tables.items():
            renamings = t.renamings
            tmp = c.schema.fields
            to_update = dict(
                (renamings[k], v) if k in renamings else (k, v)
                for k, v in tmp.items()
                if k in renamings
            )
            fields.update(to_update)
        return Schema(f'_tmp:{self.identifier}', fields)

    @property
    def renamings(self):
        r = {}
        for part in self.parts:
            if part[0] == 'rename':
                r.update(part[1][0])
            if part[0] == 'relabel':
                r.update(part[1][0])
        return r

    def _execute_pre_like(self, parent):
        like_args = self.parts[0][1]
        like_kwargs = self.parts[0][2]
        vector_index = like_kwargs['vector_index']
        like = like_args[0] if like_args else like_kwargs['like']

        similar_ids, similar_scores = self.db.get_nearest(
            like, vector_index=vector_index
        )
        similar_scores = dict(zip(similar_ids, similar_scores))
        filter_query = eval(f'table.{self.primary_id}.isin(similar_ids)')
        new_query = self.table_or_collection.filter(filter_query)

        return IbisQuery(
            db=self.db,
            identifier=self.identifier,
            parts=[
                *new_query.parts,
                *self.parts[1:],
            ],
        )

    def _execute_post_like(self, parent):
        like_args = self.parts[-1][1]
        like_kwargs = self.parts[-1][2]
        vector_index = like_kwargs['vector_index']
        like = like_args[0] if like_args else like_kwargs['like']
        [r[self.primary_id] for r in self.select_ids._execute(parent)]
        similar_ids, similar_scores = self.db.find_nearest(
            like,
            vector_index=vector_index,
            n=like_kwargs.get('n', 10),
        )
        similar_scores = dict(zip(similar_ids, similar_scores))
        output = self._execute(self[:-1].select_using_ids(similar_ids))
        output.scores = similar_scores
        return output

    def _execute_insert(self, parent):
        documents = self.documents
        for r in documents:
            if self.primary_id not in r:
                r[self.primary_id] = str(uuid.uuid4())
        ids = [r[self.primary_id] for r in documents]
        self._execute(parent, method='encode')
        return ids

    def _create_table_if_not_exists(self):
        tables = self.db.databackend.list_tables_or_collections()
        if self.identifier in tables:
            return
        self.db.databackend.create_table_and_schema(
            self.identifier,
            self.schema.raw,
        )

    def _execute(self, parent, method='encode'):
        output = super()._execute(parent, method=method)
        assert isinstance(output, pandas.DataFrame)
        output = output.to_dict(orient='records')
        component_table = self.db.tables[self.table_or_collection]
        return SuperDuperCursor(
            raw_cursor=(r for r in output),
            db=self.db,
            id_field=component_table.primary_id,
            schema=component_table.schema,
        )

    @property
    def type(self):
        return defaultdict(lambda: 'select', {'insert': 'insert'})[self.flavour]

    @property
    def primary_id(self):
        return self.db.tables[self.identifier].primary_id

    def model_update(
        self,
        ids: t.List[t.Any],
        predict_id: str,
        outputs: t.Sequence[t.Any],
        flatten: bool = False,
        **kwargs,
    ):
        if not flatten:
            return _model_update_impl(
                db=self.db,
                ids=ids,
                predict_id=predict_id,
                outputs=outputs,
                flatten=flatten,
            )
        else:
            return _model_update_impl_flatten(
                db=self.db,
                ids=ids,
                predict_id=predict_id,
                outputs=outputs,
                flatten=flatten,
            )

    def add_fold(self, fold: str):
        return self.filter(self._fold == fold)

    def select_using_ids(self, ids: t.Sequence[str]):
        filter_query = eval(f'self.{self.primary_id}.isin(ids)')
        return self.filter(filter_query)

    @property
    def select_ids(self):
        return self.select(self.primary_id)

    @applies_to('select')
    def select_ids_of_missing_outputs(self, predict_id: str):
        output_table = self.db.tables[f'_outputs.{predict_id}']
        output_table = output_table.relabel({'_base': '_outputs.' + predict_id})
        out = self.anti_join(
            output_table,
            output_table._input_id == self[self.table_or_collection.primary_id],
        )
        return out

    def select_single_id(self, id: str):
        filter_query = eval(f'table.{self.primary_id} == id')
        return self.filter(filter_query)

    @property
    def select_table(self):
        return self.table_or_collection
