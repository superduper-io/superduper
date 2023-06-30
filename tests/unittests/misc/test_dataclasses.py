from pydantic import Field
from superduperdb.misc import dataclasses as dc


@dc.dataclass
class Un:
    un: str = 'un'

    # These two unfortunately get JSONized
    nine: str = Field(default='ERROR', exclude=True)
    ten: str = dc.field(default='ERROR', repr=False, compare=False)
    eleven: dc.InitVar[str] = 'this goes up to'

    def __post_init__(self, eleven: str):
        self.seven = self.un + '-sept'
        self.eleven = eleven


UN = {
    'un': 'un',
    'nine': 'ERROR',
    'ten': 'ERROR',
}


@dc.dataclass
class Deux(Un):
    deux: str = 'deux'


@dc.dataclass
class Trois(Un):
    deux: str = 'trois'


class Objet:
    def premier(self, un: Un) -> Deux:
        return Deux(**un.dict())

    def second(self, un: Un, trois: Trois) -> Un:
        return un


@dc.dataclass
class Inclus:
    ein: Un


def test_dataclasses():
    assert Un(eleven='HAHA!').dict() == UN
    assert Inclus(Un()).asdict() == {'ein': UN}
    assert Inclus(**{'ein': UN}) == Inclus(Un())

    actual = Inclus(Un()).replace(ein=Un().replace(nine='nine')).dict()
    expected = {'ein': dict(UN, nine='nine')}
    assert actual == expected


def test_methods():
    assert [f.name for f in Un.fields()] == ['un', 'nine', 'ten']
