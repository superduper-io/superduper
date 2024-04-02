# `Trainer`

- Train a `Model` by attaching a `Trainer` component

***Usage pattern***

(Learn how to build a `Model` [here](model))

```python
from superduperdb.ext.<extension> import <ExtensionTrainer>

trainer = <ExtensionTrainer>(
    'my-trainer',
    select=train_query,   # data to use for training
    key=('X', 'y'),       # the columns/keys to use for training
    **training_params,    # can vary greatly from framework to framework
)

model = Model(
    ...     # standard arguments
    validation=validation,   # validation will be executed after training
    trainer=trainer,
)

# Applying model recognizes `.trainer` attribute
# and trains model on the `.trainer.select` attribute
db.apply(model)
```