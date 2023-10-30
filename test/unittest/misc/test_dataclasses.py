from dataclasses import InitVar, asdict, field, fields, replace

from pydantic import Field, dataclasses as dc


# TODO: Remove this because it's not a relevant test
@dc.dataclass
class Un:
    un: str = 'un'

    # These two unfortunately get JSONized
    nine: str = Field(default='ERROR', exclude=True)
    ten: str = field(default='ERROR', repr=False, compare=False)
    eleven: InitVar[str] = 'this goes up to'

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
    assert asdict(Un(eleven='HAHA!')) == UN
    assert Un(eleven='HAHA!').eleven == 'HAHA!'
    assert asdict(Inclus(Un())) == {'ein': UN}
    assert Inclus(**{'ein': UN}) == Inclus(Un())

    ein = replace(Un(), nine='nine')
    un = replace(Inclus(Un()), ein=ein)
    actual = asdict(un)
    expected = {'ein': dict(UN, nine='nine')}
    assert actual == expected


def test_methods():
    assert [f.name for f in fields(Un)] == ['un', 'nine', 'ten']
