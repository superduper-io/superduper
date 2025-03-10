import typing as t

from superduper import Component
from superduper.components.component import ensure_initialized


class CronJob(Component):
    """Run a job on a schedule.

    ***Note that this feature deploys on superduper.io Enterprise.***

    :param schedule: Schedule in cron format.
    """

    schedule: str = '0 0 * * *'

    def declare_component(self):
        """Declare component."""
        self.db.cluster.crontab.put_component(self)

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

    function: t.Callable
    identifier: str = ''

    def postinit(self):
        if not self.identifier:
            self.identifier = self.function.__name__
        return super().postinit()

    @ensure_initialized
    def run(self):
        """Run the function."""
        self.function(self.db)
