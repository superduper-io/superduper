from __future__ import annotations

import typing as t

import pandas as pd

from superduperdb.base.document import Document


class DataView(pd.DataFrame):
    """
    DataView class which extends pandas.DataFrame to provide additional functionality tailored for
    SuperDuperDB (also see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

    :param data: data to initialise the DataFrame with
    :param index: index to initialise the DataFrame with
    :param columns: columns to initialise the DataFrame with
    :param dtype: data type to initialise the DataFrame with, default None
    :param copy: whether to copy the data, default None
    """
    def __call__(self, **kwargs) -> DataView:
        """
        Quick filtering for specific data points via key word arguments. Key represents the column,
        and value represents the value to filter for.

        :param kwargs: key value pairs to filter for
        :return: DataView instance
        """
        maps = [self[k] == v for k, v in kwargs.items()]

        res = self
        for m in maps:
            res = res[m]

        return res

    @classmethod
    def from_documents(cls, documents: t.Iterable[Document]) -> DataView:
        """
        Initialises from a SuperDuperCursor.

        :param documents: iterable of documents (e.g. SuperDuperCursor) to initialise the DataView
        :return: DataView instance
        """
        return cls([doc.unpack() for doc in documents])
