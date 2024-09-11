import typing as t

from superduper import Component
from superduper.backends.base.query import Query

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class Trigger(Component):
    """Trigger a function when a condition is met.

    ***Note that this feature deploys on superduper.io Enterprise.***

    :param select: Query to select the trigger.
    """

    triggers: t.ClassVar[t.Set] = set()
    trigger: t.ClassVar[bool] = True
    type_id: t.ClassVar[str] = 'trigger'
    select: t.Union[Query, None]

    def post_create(self, db: 'Datalayer') -> None:
        super().post_create(db)

    def declare_component(self, cluster):
        super().declare_component(cluster)
        self.db.cluster.cdc.put(self)

    def trigger_ids(self, query, primary_ids):
        """Find relevant ids to trigger a function."""
        if query.table == self.select.table:
            return primary_ids
        return []
