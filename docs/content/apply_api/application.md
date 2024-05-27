# `Application`

- An `Application` ships a pre-configured functionality in a compact and easy to understand way

***Usage pattern***

(Learn how to build a model [here](model))

```python
from superduperdb import Application

template = db.load('template', 'my_template')

application = template(my_variable_1='my_value_1',
                       my_variable_2='my_value_2')

db.apply(application.copy())
```
