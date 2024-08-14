import unittest.mock as mock
from test.utils.usecase.chain_listener import build_chain_listener

from superduper import CFG


def test_output_prefix(db):
    with mock.patch.object(CFG, "output_prefix", "sddb_outputs_"):
        # Mock CFG.output_prefix
        build_chain_listener(db)
        listener_a = db.listeners["a"]
        listener_b = db.listeners["b"]
        listener_c = db.listeners["c"]

        assert listener_a.outputs == "sddb_outputs_a"
        assert listener_b.outputs == "sddb_outputs_b"
        assert listener_c.outputs == "sddb_outputs_c"

        expect_tables = [
            "documents",
            "sddb_outputs_a",
            "sddb_outputs_b",
            "sddb_outputs_c",
        ]

        tables = [
            x
            for x in db.databackend.list_tables_or_collections()
            if not x.startswith("_")
        ]

        assert set(tables) == set(expect_tables)

        outputs_a = list(listener_a.outputs_select.execute())
        assert len(outputs_a) == 6
        assert all("sddb_outputs_a" in x for x in outputs_a)

        outputs_b = list(listener_b.outputs_select.execute())
        assert len(outputs_b) == 6
        assert all("sddb_outputs_b" in x for x in outputs_b)

        outputs_c = list(listener_c.outputs_select.execute())
        assert len(outputs_c) == 6
        assert all("sddb_outputs_c" in x for x in outputs_c)
