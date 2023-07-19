# Using external data references

In an application and AI development context, certain data do not necessarily lie around in a convenient form
but may be distributed over file-servers, file-systems, web-urls, and object storage.

SuperDuperDB allows one to specify this data without first downloading and collating the data
in one place. One simply uses the ``"_content"`` keyword and inside this field specifies the
location (either file ``"file://<path>"`` or ``"http..."``) inside the ``"url"`` field:

```python
>>> db.execute(
...     coll.insert_one(
...         Document({'img': {'_content': {'url': 'https://www.superduperdb.com/logos/white.png',
...                  'type': 'image',
...                  'i': 1}}})
...     )
... )
>>> docs.find_one({'i': 1})
{'_id': ObjectId('6129008308hs99hjne05fe154a'), 
	'img': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=531x106>,
	'i': 1,
	'_fold': 'train'}
```

After adding data in this way, SuperDuperDB creates a job (read here)
which downloads the data from the specified locations using multiprocessing, and inserts the raw
bytes into the database inside the ``"_content"`` subfield.

One can see what happened under the hood by specifying ``raw=True``. The ``"bytes"`` field containing
the raw data has been added to the ``"_content"`` sub-document:

```python
>>> db.execute(coll.find_one({'i': 1}))
{'_id': ObjectId('640939c0deb71bb414652022'),
	'img': {'_content': {'url': 'https://www.superduperdb.com/logos/white.png',
											'type': 'image',
											'bytes': b'\x89PNG\r\n\x1a\n\x0...'}},
	'i': 1,
	'_fold': 'train'}
```