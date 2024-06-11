from superduperdb import Component


class CronJob(Component):
    """Run a job on a schedule.

    ***Note that this feature deploys on SuperDuperDB Enterprise.***

    :param schedule: Schedule in cron format.
    """
    schedule: str = '0 0 * * *'

    def run(self):
        raise NotImplementedError