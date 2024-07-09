import datetime
import typing as t
from abc import abstractmethod

from superduper import Component, logging


class Subscriber(Component):
    """A stream of of data which can interact with the `Datalayer`.

    ***Note that this feature deploys on superduper.io Enterprise.***

    :param table: Table to stream into
    :param key: Key to write data into
    """

    type_id: t.ClassVar[str] = "subscriber"
    table: str
    key: str

    @abstractmethod
    def next(self) -> t.Generator:
        """Next event."""
        pass

    def run(self):
        """Stream data to the database."""
        while True:
            logging.info(f'Got next item at {datetime.now().isoformat()}')
            next_items = self.next()
            next_items = [{self.key: item} for item in next_items]
            self.db[self.table].insert(next_items).execute()
