import typing as t

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer

from test.utils.usecase.chain_listener import build_chain_listener
from test.utils.usecase.graph_listener import build_graph_listener


def test_chain_listener(db: "Datalayer"):
    build_chain_listener(db)


def test_graph_listener(db: "Datalayer"):
    build_graph_listener(db)
