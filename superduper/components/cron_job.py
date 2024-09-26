import typing as t

from superduper import Component
from superduper.components.component import ensure_initialized
from superduper.components.datatype import dill_serializer


class CronJob(Component):
    """Run a job on a schedule.

    ***Note that this feature deploys on superduper.io Enterprise.***

    :param schedule: Schedule in cron format.
    """

    type_id: t.ClassVar[str] = 'cronjob'
    schedule: str = '0 0 * * *'

    def declare_component(self, cluster):
        """Declare component."""
        cluster.crontab.put(self)

    @ensure_initialized
    def run(self):
        """Run the job."""
        raise NotImplementedError


class FunctionCronJob(CronJob):
    """
    Run a function on a schedule.

    :param function: Callable to run
    """

    _artifacts = (('function', dill_serializer),)

    function: t.Callable

    @ensure_initialized
    def run(self):
        """Run the function."""
        self.function(self.db)
