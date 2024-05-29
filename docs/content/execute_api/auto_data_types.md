# Automatic data-types

A major challenge in uniting classical databases with AI, 
is that the types of data used in AI are often not supported by your database.

To solve this problem, `superduperdb` has the abstractions [`DataType`](../apply_api/datatype.md) and [`Schema`](../apply_api/schema.md).

To save developers time, by default, `superduperdb` recognizes the type of data and constructs a `Schema` based on this inference.
To learn more about setting these up manually read [the following page](./data_encodings_and_schemas.md).

## Basic usage

To learn about this feature, try these lines of code, based on sample image data we've prepared.

```bash
curl -O https://superduperdb-public-demo.s3.amazonaws.com/images.zip && unzip images.zip
```

```python
import os
import PIL.Image

from superduperdb import superduper

db = superduper('mongomock://test')

images = [PIL.Image.open(f'images/{x}') for x in os.listdir('images') if x.endswith('.png')]

# inserts the images into `db` recognizing datatypes automatically
db['images'].insert_many([{'img': img} for img in images]).execute()
```

Now if you inspect which components are available, you will see that 2 components have been added to 
the system:

```python
db.show()
```

<details>
    <summary>Outputs</summary>
    <pre>
        ```
        [{'identifier': 'pil_image', 'type_id': 'datatype'},
         {'identifier': 'AUTO:img=pil_image', 'type_id': 'schema'}]
        ```
    </pre>
</details>

To verify that the data types were correctly inferred, we can retrieve a single record.
The record is a `Document` which wraps a dictionary with important information:

```python
r = db['images'].find_one().execute()
r
```

<details>
    <summary>Outputs</summary>
    <pre>
        ```
        Document({'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=500x338 at 0x128394190>, '_fold': 'train', '_schema': 'AUTO:img=pil_image', '_id': ObjectId('6658610912e50a99219ba587')})
        ```
    </pre>
</details>


By calling the `.unpack()` method, the original data is decoded and unwrapped from the `Document`.
The result in this case is a Python `pillow` image, which may be used as direct input 
to functions from, for instance, `torchvision` or `transformers`.

```python
r.unpack()['img']
```

<details>
    <summary>Outputs</summary>
    <div>
        ![](/listening/31_0.png)
    </div>
</details>