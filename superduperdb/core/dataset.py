import datetime
from functools import cached_property

import numpy

from superduperdb.core.base import Component, DBPlaceholder
from superduperdb.datalayer.base.query import Select

import typing as t


class Dataset(Component):
    variety = 'dataset'
    repopulate_on_init = True

    def __init__(
        self,
        identifier: str,
        select: Select,
        sample_size: t.Optional[int] = None,
        random_seed: t.Optional[int] = None,
    ):
        super().__init__(identifier)

        self.select = select
        self.database = DBPlaceholder()
        self.creation_date = datetime.datetime.now()
        self.sample_size = sample_size
        self.random_seed = random_seed

    @cached_property
    def random(self):
        return numpy.random.default_rng(seed=self.random_seed)

    def _post_attach_database(self):
        data = list(self.database.execute(self.select))
        if self.sample_size is not None and self.sample_size < len(data):
            perm = self.random.permutation(len(data)).tolist()
            data = [data[perm[i]] for i in range(self.sample_size)]
        self.data = data

    def asdict(self):
        return {
            'identifier': self.identifier,
            'select': self.select.dict(),
            'creation_date': self.creation_date,
        }
