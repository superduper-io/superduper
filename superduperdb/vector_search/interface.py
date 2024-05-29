import typing as t

import numpy as np

from superduperdb import CFG
from superduperdb.misc.server import request_server
from superduperdb.vector_search.base import BaseVectorSearcher, VectorItem

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


class FastVectorSearcher(BaseVectorSearcher):
    """Fast vector searcher implementation using the server.

    :param db: Datalayer instance
    :param vector_searcher: Vector searcher instance
    :param vector_index: Vector index name
    """

    def __init__(self, db: 'Datalayer', vector_searcher, vector_index: str):
        self.searcher = vector_searcher
        self.vector_index = vector_index

        if CFG.cluster.vector_search.uri is not None:
            if not db.server_mode:
                request_server(
                    service='vector_search',
                    endpoint='create/search',
                    args={
                        'vector_index': self.vector_index,
                    },
                    type='get',
                )

    @staticmethod
    def drop_remote(index):
        """Drop a vector index from the remote.

        :param index: The index to drop.
        """
        request_server(
            service='vector_search',
            endpoint='drop/search',
            args={
                'vector_index': index,
            },
        )
        return

    def drop(self):
        """Drop the vector index from the remote."""
        if CFG.cluster.vector_search.uri is not None:
            self.drop_remote(self.vector_index)
            request_server

    def __len__(self):
        return len(self.searcher)

    def add(self, items: t.Sequence[VectorItem]) -> None:
        """
        Add items to the index.

        :param items: t.Sequence of VectorItems
        """
        vector_items = [{'vector': i.vector, 'id': i.id} for i in items]
        if CFG.cluster.vector_search.uri is not None:
            request_server(
                service='vector_search',
                data=vector_items,
                endpoint='add/search',
                args={
                    'vector_index': self.vector_index,
                },
            )
            return

        return self.searcher.add(items)

    def delete(self, ids: t.Sequence[str]) -> None:
        """Remove items from the index.

        :param ids: t.Sequence of ids of vectors.
        """
        if CFG.cluster.vector_search.uri is not None:
            request_server(
                service='vector_search',
                data=ids,
                endpoint='delete/search',
                args={
                    'vector_index': self.vector_index,
                },
            )
            return

        return self.searcher.delete(ids)

    def find_nearest_from_id(
        self,
        _id,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """
        Find the nearest vectors to the vector with the given id.

        :param _id: id of the vector
        :param n: number of nearest vectors to return
        :param within_ids: list of ids to search within
        """
        if CFG.cluster.vector_search.uri is not None:
            response = request_server(
                service='vector_search',
                endpoint='query/id/search',
                args={'vector_index': self.vector_index, 'n': n, 'id': _id},
            )
            return response['ids'], response['scores']

        return self.searcher.find_nearest_from_id(_id, n=n, within_ids=within_ids)

    def find_nearest_from_array(
        self,
        h: np.typing.ArrayLike,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """
        Find the nearest vectors to the given vector.

        :param h: vector
        :param n: number of nearest vectors to return
        :param within_ids: list of ids to search within
        """
        if CFG.cluster.vector_search.uri is not None:
            response = request_server(
                service='vector_search',
                data=h,
                endpoint='query/search',
                args={'vector_index': self.vector_index, 'n': n},
            )
            return response['ids'], response['scores']

        return self.searcher.find_nearest_from_array(h=h, n=n, within_ids=within_ids)

    def post_create(self):
        """Post create method for vector searcher."""
        if CFG.cluster.is_remote_vector_search:
            request_server(
                service='vector_search',
                endpoint='query/post_create',
                args={'vector_index': self.vector_index},
                type='get',
            )
            return
        self.searcher.post_create()
