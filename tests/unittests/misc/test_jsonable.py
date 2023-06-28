import json
from overrides import override
from superduperdb.misc.jsonable import Box


class Database:
    def __init__(self, name: str):
        self.__name = name

    @property
    def name(self):
        return self.__name


class DatabaseBox(Box[Database]):
    db_name: str

    @classmethod
    def box(cls, contents: Database) -> 'DatabaseBox':
        # Crave covariant return types
        return super().box(contents)  # type: ignore[return-value]

    @override
    def _box_to_contents(self) -> Database:
        return Database(self.db_name)

    @classmethod
    def _contents_to_box(cls, db: Database) -> 'DatabaseBox[Database]':
        return cls(db_name=db.name)


def test_box():
    db = Database('test-db')
    box = DatabaseBox.box(db)

    j = json.dumps(box.dict())
    assert j == '{"db_name": "test-db"}'

    box2 = DatabaseBox(**json.loads(j))
    assert box2().name == db.name
