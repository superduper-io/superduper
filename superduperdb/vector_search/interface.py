import typing as t

import numpy as np

from superduperdb import CFG
from superduperdb.base import exceptions
from superduperdb.misc.server import request_server
from superduperdb.vector_search.base import BaseVectorSearcher, VectorItem

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


class FastVectorSearcher(BaseVectorSearcher):
    def __init__(self, db: 'Datalayer', vector_searcher, vector_index: str):
        self.searcher = vector_searcher
        self.vector_index = vector_index

        if CFG.cluster.vector_search:
            if not db.server_mode:
                request_server(
                    service='vector_search',
                    endpoint='create/search',
                    args={
                        'vector_index': self.vector_index,
                    },
                    type='get',
                )

    def __len__(self):
        return len(self.searcher)

    def add(self, items: t.Sequence[VectorItem]) -> None:
        """
        Add items to the index.

        :param items: t.Sequence of VectorItems
        """
        try:
            vector_items = [{'vector': i.vector, 'id': i.id} for i in items]
            if CFG.cluster.vector_search:
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
        except Exception as e:
            local_msg = (
                'remote vector search service'
                if CFG.cluster.vector_search
                else 'local vector search'
            )
            raise exceptions.VectorSearchException(
                f'Error while adding vector to {local_msg}'
            ) from e

    def delete(self, ids: t.Sequence[str]) -> None:
        """
        Remove items from the index

        :param ids: t.Sequence of ids of vectors.
        """
        try:
            if CFG.cluster.vector_search:
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
        except Exception as e:
            local_msg = (
                'remote vector search service'
                if CFG.cluster.vector_search
                else 'local vector search'
            )
            raise exceptions.VectorSearchException(
                f'Error while deleting ids {ids} from {local_msg}'
            ) from e

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
        """
        try:
            if CFG.cluster.vector_search:
                response = request_server(
                    service='vector_search',
                    endpoint='query/id/search',
                    args={'vector_index': self.vector_index, 'n': n, 'id': _id},
                )
                return response['ids'], response['scores']

            return self.searcher.find_nearest_from_id(_id, n=n, within_ids=within_ids)
        except Exception as e:
            local_msg = (
                'remote vector search service'
                if CFG.cluster.vector_search
                else 'local vector search'
            )
            raise exceptions.VectorSearchException(
                f'Error while finding nearest array from {local_msg} \n\
                The problem might be either wrong id {_id} provided or vector database \
                is empty (Not initialized properly), check if model/listener outputs \
                are successfully computed, generally have issues \
                when computes is distributed to dask for example. '
            ) from e

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
        """
        try:
            if CFG.cluster.vector_search:
                response = request_server(
                    service='vector_search',
                    data=h,
                    endpoint='query/search',
                    args={'vector_index': self.vector_index, 'n': n},
                )
                return response['ids'], response['scores']

            return self.searcher.find_nearest_from_array(
                h=h, n=n, within_ids=within_ids
            )
        except Exception as e:
            local_msg = (
                'remote vector search service'
                if CFG.cluster.vector_search
                else 'local vector search'
            )
            raise exceptions.VectorSearchException(
                f'Error while finding nearest array from {local_msg} \n\
                The problem might be either wrong vector provided or vector database \
                is empty (Not initialized properly), check if model/listener outputs \
                are successfully computed, generally have issues \
                when computes is distributed to dask for example. '
            ) from e
