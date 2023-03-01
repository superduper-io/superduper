Adding content to SuperDuperDB
==============================

Many AI models may be used with interesting content such as image, video or audio, to name but a few types.
MongoDB does not support these types of data natively.
Accordingly, SuperDuperDB allows users to use user-defined :ref:`types <Types in SuperDuperDB>`
to support interesting content.

The most straight-forward way to do this, is to define a types which includes the ``types`` attribute; see 
:ref:`here <Types in SuperDuperDB>` for an example. If the types of certain subfields of added documents
are of one of these types, then SuperDuperDB may be used to add this types in a way directly analogous to 
a standard MongoDB update:

.. code-block:: python

	>>> import requests, PIL.Image, io
	>>> bytes_ = requests.get('https://www.superduperdb.com/logos/white.png').content
	>>> image = PIL.Image.open(io.BytesIO(bytes_))
	>>> docs.insert_one({'img': image, 'i': 0)
	>>> docs.find_one({'i': 0})
	{'_id': ObjectId('63fca4325d2a192e05fe154a'), 
	 'img': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=531x106>,
	 '_fold': 'train'}

If one hasn't set the ``types`` attribute, one may add the content more explicitly like this

.. code-block:: python

	>>> docs.insert_one({'img': {'_content': {'bytes': bytes_, 'type': 'image'}}, 'i': 0})

It's important to add the ``"type"`` key-word to the ``"_content"`` subfield, so that 
SuperDuperDB knows how to handle the content.

One can also specify the content from a URL - SuperDuperDB makes sure that the ``bytes`` at that URL
are added to the database. 

.. code-block:: python

	>>> docs.insert_one({'img': {'_content': {'url': 'https://www.superduperdb.com/logos/white.png',
	...                  'type': 'image',
	...                  'i': 1}})
	>>> docs.find_one({'i': 1})
	{'_id': ObjectId('6129008308hs99hjne05fe154a'), 
	 'img': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=531x106>,
	 '_fold': 'train'}

One can retrieve the ``bytes`` in raw form from the data base by specifying ``raw=True``:

.. code-block:: python
	
	>>> docs.find_one({'i': 1}, raw=True)['img']['_content']['bytes']
	b'\x1c\xf4\xd0;[\xe4\xb4;D)f\xbc\x7f\xfc\x83\xb9e\xf2\x87;:H\x06;q\x95...'

