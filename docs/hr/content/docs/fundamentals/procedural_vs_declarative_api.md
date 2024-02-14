---
sidebar_position: 6
---

# Procedural and Declarative API

`superduperdb` provides 2 principle approaches for applying AI to the database

| Method      | Description                                            | Example                                                     |
|-------------|--------------------------------------------------------|-------------------------------------------------------------| 
| Procedural  | *Telling SuperDuperDB **what** to do*                  | `model.predict(X='txt', db=db)`                             |
| Declarative | *Telling SuperDuperDB which **state** it should be in* | `db.add(Listener(model=model, key='txt', select=c.find()))` |

### Procedural API

The procedural API is better suited for experimentation, and will be more familiar to 
data-scientists, where the procedural API is inspired by the API of well known 
PyData packages, such as `sklearn`. 

In particular, much functionality is covered by:

- `Model.predict`
- `Model.fit`

### Declarative API

The declarative API is designed for production use-cases, and more sophisticated workflows.
It is better suited for linking dependencies between components, and allows users 
to build stacks of functionality, with unlimited complexity. The declarative API
will be more familiar to developers with an engineering or infrastructure background.

With the declarative API, developers work with:

```python
db.add(<component-to-be-added>)
```

The operand of this function call is always an instance of a descendant of `Component`, but may also
contain itself many other `Component` descendant instances.

For instance, creating a `VectorIndex` involves also 
creating a `Listener` and a `Model` inline.

```python
db.add(
    VectorIndex(
        'my-index'
        indexing_listener=Listener(
            model=model,
            key='txt',
            select=my_collection.find(),
        ),
    )
)
```

Read more about the `VectorIndex` concept [here](../walkthrough/vector_search.md).
