import dataclasses as dc
import re
import typing as t

from superduperdb.db.base.cursor import SuperDuperCursor


@dc.dataclass
class SuperDuperIbisCursor(SuperDuperCursor):
    db: t.Any = None

    def _get_encoders_for_schema(self, schema_names):
        encoders = {}
        for name in schema_names:
            try:
                # TODO use version
                identifier, version = re.match(
                    '.*::_encodable=(.*)/(.*)::', name
                ).groups()[:2]
            except AttributeError as e:
                if "'NoneType' object has no attribute 'groups'" in str(e):
                    continue
                raise e
            name = name.split('::_encodable')[0]
            encoders[name] = self.encoders[identifier]
        return encoders

    def execute(self):
        if self.raw_cursor[0] is None:
            return

        schema_names = self.raw_cursor.schema().names

        self.raw_cursor = self.raw_cursor.execute()
        self.encoders = self._get_encoders_for_schema(schema_names)

        if self.encoders:
            new_cols = [c.split('::_encodable')[0] for c in self.raw_cursor.columns]
            self.raw_cursor.columns = new_cols

        self.dict_cursor = self.raw_cursor.to_dict(orient="records")
        self._n = len(self.dict_cursor)
        self._index = 0
        return self

    def cursor_next(self):
        if self._index < self._n:
            row = self.dict_cursor[self._index]
            self._index += 1
            return row
        else:
            raise StopIteration
