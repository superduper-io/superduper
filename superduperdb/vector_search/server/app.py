import typing as t

from pydantic import BaseModel

from superduperdb import CFG
from superduperdb.base.datalayer import Datalayer
from superduperdb.server.app import DatalayerDependency, SuperDuperApp
from superduperdb.vector_search.server import service

assert isinstance(
    CFG.cluster.vector_search, str
), "cluster.vector_search should be set with a valid uri"

port = int(CFG.cluster.vector_search.split(':')[-1])
app = SuperDuperApp('vector_search', port=port)


class VectorItem(BaseModel):
    id: str
    vector: service.ListVectorType


@app.add("/create/search", status_code=200, method='get')
def create_search(vector_index: str, db: Datalayer = DatalayerDependency()):
    service.create_search(vector_index=vector_index, db=db)
    return {'message': 'Vector index created successfully'}


@app.add("/query/id/search", method='post')
def query_search_by_id(
    id: str, vector_index: str, n: int = 100, db: Datalayer = DatalayerDependency()
):
    ids, scores = service.query_search_from_id(
        id, vector_index=vector_index, n=n, db=db
    )
    return {'ids': ids, 'scores': scores}


@app.add("/query/search")
def query_search_by_array(
    vector: service.ListVectorType,
    vector_index: str,
    n: int = 100,
    db: Datalayer = DatalayerDependency(),
):
    ids, scores = service.query_search_from_array(
        vector, vector_index=vector_index, n=n, db=db
    )
    return {'ids': ids, 'scores': scores}


@app.add("/add/search")
def add_search(
    vectors: t.List[VectorItem],
    vector_index: str,
    db: Datalayer = DatalayerDependency(),
):
    service.add_search(vectors, vector_index=vector_index, db=db)
    return {'message': 'Added vectors successfully'}


@app.add("/delete/search")
def delete_search(
    ids: t.List[str], vector_index: str, db: Datalayer = DatalayerDependency()
):
    service.delete_search(ids, vector_index=vector_index, db=db)
    return {'message': 'Ids deleted successfully'}


@app.add("/list/search")
def list_search(db: Datalayer = DatalayerDependency()):
    return service.list_search(db)
