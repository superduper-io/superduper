from superduperdb import CFG
from superduperdb.base.datalayer import Datalayer
from superduperdb.components.listener import Listener
from superduperdb.server import app as superduperapp

assert isinstance(CFG.cluster.cdc, str), "cluster.cdc should be set with a valid uri"
port = int(CFG.cluster.cdc.split(':')[-1])
app = superduperapp.SuperDuperApp('cdc', port=port)


@app.startup
def cdc_startup(db: Datalayer):
    db.cdc.start()


@app.add('/listener/add', method='get')
def add_listener(name: str, db: Datalayer = superduperapp.DatalayerDependency()):
    """
    Endpoint for adding a listener to cdc
    """
    listener = db.load('listener', name)
    assert isinstance(listener, Listener)
    db.cdc.add(listener)


@app.add('/listener/delete', method='get')
def remove_listener(name: str, db: Datalayer = superduperapp.DatalayerDependency()):
    """
    Endpoint for removing a listener from cdc
    """
    listener = db.load('listener', name)
    assert isinstance(listener, Listener)
    on = listener.select.table_or_collection.identifier
    db.cdc.stop(on)  # type: ignore[arg-type]
