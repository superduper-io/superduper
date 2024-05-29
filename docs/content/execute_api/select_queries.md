# Selecting data

Selecting data involves building a compositional query 
staring with a table of collection, and repeatingly calling
methods to build a complex query:

```python
q = db['table_name'].method_1(*args_1, **kwargs_1).method_2(*args_2, **kwargs_2)....
```

As usual, the query is executed with:

```
q.execute()
```

## Read more

```mdx-code-block
import DocCardList from '@theme/DocCardList';

<DocCardList />
```