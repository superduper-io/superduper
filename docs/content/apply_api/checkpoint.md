# `Checkpoint`

- Save intermediate results of training via SuperDuperDB
- Load a different point of the training process by specifying `Checkpoint` explicitly
- Particularly useful with deep-learning models

***Usage pattern***

```python
from superduperdb import Model
from superduperdb.components.training import Checkpoint

class MyModel(Model):
    checkpoints: t.List[Checkpoint]
    best_checkpoint: t.Optional[int] = None

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)

        if self.best_checkpoint is not None:
            self.load_weights(self.checkpoints[self.best_checkpoint])

    def load_weights(self):
        ... # custom load logic

my_model = MyModel('my-model')

my_model.checkpoints.append('path/to/model/weights-0.pt')
my_model.checkpoints.append('path/to/model/weights-1.pt')
my_model.best_checkpoint = 1

# saves model as well as checkpoints to db.artifact_store
db.apply(my_model)     

# loads `self.checkpoints[1]`
m = db.load('model', 'my-model')
```