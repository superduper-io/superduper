from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore

from superduperdb import logging


class Daemon:
    def __init__(self, db):
        self.db = db
        self.scheduler = BlockingScheduler(
            jobstores={'default': MemoryJobStore()},
            executors={'default': ThreadPoolExecutor(20)},
        )
        self.jobs = {}
        for job in db.show('cronjob'):
            self.add_job(job, db)

    def add_job(self, job):
        logging.info('Adding job {}'.format(job))
        job = db.load('cronjob', job)
        self.jobs[job.identifier] = job.uuid
        self._add_job_to_scheduler(job)

    def _add_job_to_scheduler(self, job):
        minute, hour, day, month, day_of_week = job.schedule.split()
        self.scheduler.add_job(
            job.run,
            'cron',
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            id=job.uuid,
        )

    def remove_job(self, identifier):
        self.scheduler.remove_job(self.jobs[identifier])

    def list_jobs(self):
        return self.jobs

    def start(self):
        logging.info('Starting scheduler')
        self.scheduler.start()


if __name__ == '__main__':
    from superduperdb import superduper
    db = superduper()
    Daemon(db).start()