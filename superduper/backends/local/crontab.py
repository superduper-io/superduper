import typing as t

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from superduper import logging
from superduper.backends.base.crontab import CrontabBackend

if t.TYPE_CHECKING:
    from superduper import Component


class LocalCrontabBackend(CrontabBackend):
    """Local crontab backend."""

    def __init__(self):
        super().__init__()
        self.jobs = set()
        self._job_uuids = set()
        self.scheduler = BackgroundScheduler(
            jobstores={"default": MemoryJobStore()},
            executors={"default": ThreadPoolExecutor(20)},
        )

    def _add_job_to_scheduler(self, job):
        minute, hour, day, month, day_of_week = job.schedule.split()
        self.scheduler.add_job(
            job.run,
            "cron",
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            id=job.uuid,
        )
        # check if scheduler is not started
        if not self.scheduler.running:
            logging.info("Starting scheduler")
            self.scheduler.start()

    def _put(self, item):
        from superduper.components.cron_job import CronJob

        assert isinstance(item, CronJob)
        self.jobs.add((item.type_id, item.identifier))
        self._job_uuids.add(item.uuid)
        self._add_job_to_scheduler(item)

    def list(self):
        """List crontab items."""
        raise NotImplementedError

    def __delitem__(self, item):
        raise NotImplementedError

    def list_components(self):
        """List components."""
        return list(self.jobs)

    def list_uuids(self):
        """List UUIDs of components."""
        return list(self._job_uuids)

    def drop_component(self, uuid: str):
        """Drop the crontab.

        :param uuid: Component uuid to remove.
        """
        self.scheduler.remove_job(uuid)

    def drop(self, component: t.Optional['Component'] = None):
        """Drop the crontab.

        :param component: Component to remove.
        """
        if component:
            self.scheduler.remove_job(component.uuid)
        else:
            for job_id in self._job_uuids:
                self.scheduler.remove_job(job_id)

    def initialize(self):
        """Initialize the crontab."""
        for component_data in self.db.show():
            type_id = component_data['type_id']
            identifier = component_data['identifier']
            r = self.db.show(type_id=type_id, identifier=identifier, version=-1)
            if r.get('schedule'):
                obj = self.db.load(type_id=type_id, identifier=identifier)
                from superduper.components.cron_job import CronJob

                if isinstance(obj, CronJob):
                    self.put(obj)
