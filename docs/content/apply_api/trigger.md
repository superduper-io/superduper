# Trigger

- Listen for update, inserts and deletes
- Take a specific action contigent on these changes
- Can be deployed on Superduper Enterprise

***Usage pattern***

```python
from superduper.components.trigger import Trigger

class MyTrigger(Trigger):
    def if_change(self, ids):
        data = db[self.table].select_ids(ids).execute()
        for r in data:
            if 'urgent' in r['title']:
                db['notifications'].insert_one({
                    'status': 'urgent',
                    'msg': r['msg'],
                }).execute()

my_trigger = MyTrigger('urgent', on='insert')
```