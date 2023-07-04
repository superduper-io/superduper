import superduperdb as s
import typing as t
from abc import ABC, abstractmethod


class SelectOne(s.JSONable, ABC):
    """
    Base class for queries which return a single line/ record of data
    """

    @abstractmethod
    def __call__(self, db):
        pass


class Like(s.JSONable, ABC):
    """
    Base class for queries which invoke vector-search
    """

    @abstractmethod
    def __call__(self, db):
        pass


class Select(s.JSONable, ABC):
    """
    Abstract base class, encapsulating Select database queries/ datalayer reads.
    This allows the concrete implementation of each datalayer to differ substantially on
    stored properties necessary for querying the DB.
    """

    @property
    @abstractmethod
    def select_table(self):
        pass

    @abstractmethod
    def is_trivial(self) -> bool:
        # Determines when a select statement is "just" select everything.
        # For example, in SQL: "FROM my_table SELECT *"
        # For example, in MongoDB: "collection.find()"
        pass

    @property
    @abstractmethod
    def select_ids(self) -> 'Select':
        # Converts the query into a query which only returns the id
        # of each column/ document.
        pass

    @abstractmethod
    def select_using_ids(
        self,
        ids: t.List[str],
    ) -> t.Any:
        """
        Create a select using the same query, subset to the specified ids

        :param ids: string ids to which subsetting should occur
        """
        # Converts the query into a query which sub-selects from the ids specified.
        pass

    @abstractmethod
    def add_fold(self, fold: str) -> 'Select':
        """
        Create a select which selects the same data, but additionally restricts to the
        fold specified

        :param fold: possible values {'train', 'valid'}
        """
        # Converts the query into a query which sub-selects based on the specified
        # tag "fold"
        pass

    @abstractmethod
    def model_update(self, db, model, key, outputs, ids):
        """
        Add outputs of ``model`` to the datalayer ``db``.

        :param db: datalayer
        :param model: model identifier to be updated against
        :param key: key on which model was applied
        :param outputs: (encoded) outputs to be added
        :param ids: ids of input documents corresponding to each output
        """
        pass

    @abstractmethod
    def __call__(self, db):
        """
        Apply query to datalayer

        :param db: datalayer instance
        """
        pass


class Insert(s.JSONable, ABC):
    """
    Base class for database inserts.

    :param refresh: toggle to ``False`` to suppress job triggering
                    (model computations on new docs)
    :param verbose: toggle tp ``False`` to suppress/reduce stdout
    :param documents: list of documents to insert

    """

    # must implement attribute/ property self.documents
    refresh: bool = True
    verbose: bool = True
    documents: t.List

    @property
    @abstractmethod
    def table(self):
        # extracts the table collection from the object
        pass

    @property
    @abstractmethod
    def select_table(self) -> Select:
        # returns a Select object which selects the table into which the insert
        # was inserted
        pass

    @abstractmethod
    def __call__(self, db):
        """
        Apply query to datalayer

        :param db: datalayer instance
        """
        pass


class Delete(s.JSONable, ABC):
    """
    Base class for deleting documents from datalayer
    """

    @abstractmethod
    def __call__(self, db):
        """
        Apply query to datalayer

        :param db: datalayer instance
        """
        pass


class Update(s.JSONable, ABC):
    """
    Base class for database updates.

    :param refresh: toggle to ``False`` to suppress job triggering
                    (model computations on new docs)
    :param verbose: toggle tp ``False`` to suppress/reduce stdout

    """

    refresh: bool = True
    verbose: bool = True

    @property
    @abstractmethod
    def select_table(self):
        pass

    @property
    @abstractmethod
    def select(self):
        """
        Converts the update object to a Select object, which selects where
        the update was made.
        """
        pass

    @property
    @abstractmethod
    def select_ids(self):
        """
        Converts the update object to a Select object, which selects where
        the update was made, and returns only ids.
        """
        pass

    @abstractmethod
    def __call__(self, db):
        """
        Apply query to datalayer.

        :param db: datalayer instance
        """
        pass
