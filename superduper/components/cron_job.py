import typing as t

from superduper import Component, logging
from superduper.components.component import ensure_setup


class CronJob(Component):
    """Run a job on a schedule.

    ***Note that this feature deploys on superduper.io Enterprise.***

    :param schedule: Schedule in cron format.
    """

    schedule: str = '0 0 * * *'

    def on_create(self):
        """Declare component."""
        self.db.cluster.crontab.put_component(self)

    @ensure_setup
    def run(self):
        """Run the job."""
        raise NotImplementedError

    def cleanup(self):
        """Cleanup crontab service."""
        super().cleanup()
        self.db.cluster.crontab.drop_component(self.component, self.identifier)

    def initialize(self):
        """Initialize the crontab."""
        self.db.cluster.crontab.put_component(self)


class FunctionCronJob(CronJob):
    """
    Run a function on a schedule.

    :param function: Callable to run
    """

    function: t.Callable
    identifier: str = ''

    def postinit(self):
        """Post initialization method."""
        if not self.identifier:
            self.identifier = self.function.__name__
        return super().postinit()

    @ensure_setup
    def run(self):
        """Run the function."""
        logging.info(f"Running cron job {self.identifier} with {self.function}")
        self.function(self.db)
        logging.info(f"Running cron job {self.identifier} with {self.function}... DONE")
