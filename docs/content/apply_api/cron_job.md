# `Cron Job`

- Iterate computations, queries and actions on a crontab
- Can be deployed on SuperDuperDB Enterprise

***Usage pattern***

Cron-job can take any actions relating to `db`
which is loaded as an attribute of the `Component`.

```python
import datetime
from superduperdb.components.cron_job import CronJob

class MyCronJob(CronJob):
    table: str

    # overwriting this function defines actions to be 
    # taken on a schedule
    def run(self):
        results = list(self.db[self.table].select())

        date = str(datetime.now())

        with open(f'{date}.bak', 'wb') as f:
            json.dump(results)
        
        # for example, backing up a collection every day
        os.system(f'aws s3 cp {date}.bak s3://my-bucket/{date}.bak')

cron_job = MyCronJob(table='documents', schedule='0 0 * * * *')

db.apply(cron_job)
```