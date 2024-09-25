<!-- Auto-generated content start -->
# superduper_pillow

SuperDuper Pillow is a plugin for SuperDuper that provides support for Pillow.

## Installation

```bash
pip install superduper_pillow
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/pillow)
- [API-docs](/docs/api/plugins/superduper_pillow)

| Class | Description |
|---|---|



<!-- Auto-generated content end -->

<!-- Add your additional content below -->

## Examples

We can use the `pil_image` field type to store images in a database.

```python
from superduper import superduper
from superduper import Table, Schema
from superduper_pillow import pil_image
from PIL import Image

table = Table("image", schema=Schema(identifier="image", fields={"img": pil_image}))
db = superduper('mongomock://test')
db.apply(table)

# Inserting an image
db["image"].insert([{"img": Image.open("test/material/data/1x1.png")}]).execute()

# Selecting an image
list(db["image"].select().execute())
```


