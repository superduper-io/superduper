# Data inserts

SuperDuperDB allows developers to insert data from a variety of sources, 
encoding and decoding objects, such as images and videos, not usually handled 
explicitly by the `db.databackend`.

The philosophy, however, is to give a very similar developer experience 
to querying your `db.databackend` natively, but returning exactly the objects
expected by your AI models and components.

For example if images have been "inserted" to a collection in MongoDB, 
the following cursor can very easily be used as a data sampler, loading
`PIL.Image` objects:

```python
cursor = db.execute(images.find())
```

## Read more

```mdx-code-block
import DocCardList from '@theme/DocCardList';

<DocCardList />
```