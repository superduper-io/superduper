import typing as t

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from superduper import logging
from superduper.backends.base.backends import Bookkeeping
from superduper.backends.base.crontab import CrontabBackend
from superduper.components.cron_job import CronJob


class JobWrapper:
    """Job wrapper.

    :param job: CronJob component.
    :param scheduler: LocalScheduler instance.
    """

    def __init__(self, job, scheduler, on_compute: bool = False):
        self.job = job
        self.scheduler = scheduler
        self.on_compute = on_compute

    @property
    def identifier(self):
        return self.job.uuid

    def drop(self):
        """Drop the job."""
        if self.scheduler.get_job(self.job.uuid):
            logging.info(f"Removing job {self.job.uuid}")
            self.scheduler.remove_job(self.job.uuid)
        else:
            logging.info(f"Job {self.job.uuid} not found, skipping removal.")

    def initialize(self):
        """Initialize the job."""
        minute, hour, day, month, day_of_week = self.job.schedule.split()
        # check if job already exists
        if not self.scheduler.get_job(self.job.uuid):
            self.scheduler.add_job(
                (
                    self.job.run_and_propagate_failure
                    if not self.on_compute
                    else self.job.run_on_compute
                ),
                "cron",
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week,
                id=self.job.uuid,
            )
        else:
            logging.info(f"Job {self.job.uuid} already exists, skipping creation.")
        # check if scheduler is not started
        if not self.scheduler.running:
            logging.info("Starting scheduler")
            self.scheduler.start()


class LocalCrontabBackend(Bookkeeping, CrontabBackend):
    """Local crontab backend."""

    cls = CronJob

    def __init__(self, on_compute: bool = False, compute_kwargs: t.Dict | None = None):
        Bookkeeping.__init__(self)
        CrontabBackend.__init__(self)
        self.on_compute = on_compute
        self.compute_kwargs = compute_kwargs
        self.scheduler = BackgroundScheduler(
            jobstores={"default": MemoryJobStore()},
            executors={"default": ThreadPoolExecutor(20)},
        )

    def build_tool(self, component: 'CronJob'):
        if component.compute_kwargs is None:
            component.compute_kwargs = {}
        if self.compute_kwargs is not None:
            component.compute_kwargs.update(self.compute_kwargs)
        return JobWrapper(component, self.scheduler, self.on_compute)

    def initialize(self):
        """Initialize the crontab."""
        self.initialize_with_components()
