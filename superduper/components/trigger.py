import typing as t

from superduper import Component, logging
from superduper.backends.base.query import Query


class Trigger(Component):
    """Trigger a function when a condition is met.

    ***Note that this feature deploys on superduper.io Enterprise.***

    :param select: Query to select the trigger.
    """
    type_id: t.ClassVar[str] = 'trigger'
    select: t.Union[Query, None]

    def trigger_ids(self, query, primary_ids):
        if query.table == self.select.table:
            query: Query = query.select_using_ids(primary_ids)
            query = query.select_ids
            return [r[query.primary_id] for r in query.execute()]
        return []