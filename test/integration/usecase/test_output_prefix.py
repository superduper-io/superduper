import unittest.mock as mock
from test.utils.usecase.chain_listener import build_chain_listener

from superduper import CFG


def test_output_prefix(db):
    with mock.patch.object(CFG, "output_prefix", "sddb_outputs_"):
        # Mock CFG.output_prefix
        build_chain_listener(db)
        listener_a = db.load('Listener', 'a')
        listener_b = db.load('Listener', 'b')
        listener_c = db.load('Listener', 'c')

        assert listener_a.outputs.startswith("sddb_outputs_a")
        assert listener_b.outputs.startswith("sddb_outputs_b")
        assert listener_c.outputs.startswith("sddb_outputs_c")

        expect_tables = [
            "documents",
            "sddb_outputs_a",
            "sddb_outputs_b",
            "sddb_outputs_c",
        ]

        tables = [x for x in db.databackend.list_tables() if not x.startswith("_")]
        for t in expect_tables:
            assert any(k.startswith(t) for k in tables)

        outputs_a = db[listener_a.outputs].select().execute()
        assert len(outputs_a) == 6
        for r in outputs_a:
            assert any(k.startswith("sddb_outputs_a") for k in r)

        outputs_b = db[listener_b.outputs].select().execute()
        assert len(outputs_b) == 6
        for r in outputs_b:
            assert any(k.startswith("sddb_outputs_b") for k in r)

        outputs_c = db[listener_c.outputs].select().execute()
        assert len(outputs_c) == 6
        for r in outputs_c:
            assert any(k.startswith("sddb_outputs_c") for k in r)
