import typing as t

from superduper import Component
from superduper.components.component import ensure_initialized


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

    def cleanup(self, db):
        """Cleanup crontab service.

        :param db: Database instance.
        """
        super().cleanup(db=db)
        db.cluster.crontab.drop(self)


class FunctionCronJob(CronJob):
    """
    Run a function on a schedule.

    :param function: Callable to run
    """

    _fields = {'function': 'default'}

    function: t.Callable

    @ensure_initialized
    def run(self):
        """Run the function."""
        self.function(self.db)
