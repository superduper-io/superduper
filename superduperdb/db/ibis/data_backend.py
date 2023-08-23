import click
from ibis.backends.base import BaseBackend

from superduperdb.db.base.data_backend import BaseDataBackend
from superduperdb.misc.colors import Colors


class IbisDataBackend(BaseDataBackend):

    def __init__(self, conn: BaseBackend , name:str):
        super().__init__(conn=conn, name=name)
        self._db = self.conn

    @property
    def db(self):
        return self._db

    def drop(self, force: bool = False):
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop the data-backend? ',
                default=False,
            ):
                print('Aborting...')
        # TODO: Implement drop functionality
        return
