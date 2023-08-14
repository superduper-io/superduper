(externaldata)=
# How SuperDuperDB handles external data

SuperDuperDB has 2 handy features which work in concert, which make working with data and AI together
much more convenient:

- Store Python objects which aren't serializable in standard data handled by JSON.
- Add data by reference to web-content, `aws s3` URIs and local filesystems

The key abstraction for doing this is `superduperdb.core.encoder.Encoder`:

An `Encoder` instance attaches an `encoder` and `decoder` to the data to be sent to the database. 
These are wrapped in an `Encodable` instance:

```python
>>> from superduperdb.core.encoder import Encoder
>>> import torch
>>> x = torch.randn(10)
# `encoder` and `decoder` handle turning an object into bytes and back
>>> enc = Encoder('my-enc', encoder=my_encoder, decoder=my_decoder)
>>> e = enc(x)
>>> type(e)
<class 'superduperdb.core.encoder.Encodable'>
```

The `Encodable` instance then has the information it needs to send the data to the database:

```python
>>> e.encode()
{'_content': {'bytes': b'\x80\x04\x95\xac\x01\x00\x00\x00\x00\x00\x00\x8c\x0ctorch._utils\x94\x8c\x12_rebuild_tensor_v2\x94\x93\x94(\x8c\rtorch.storage\x94\x8c\x10_load_from_bytes\x94\x93\x94B%\x01\x00\x00\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9\x03.\x80\x02}q\x00(X\x10\x00\x00\x00protocol_versionq\x01M\xe9\x03X\r\x00\x00\x00little_endianq\x02\x88X\n\x00\x00\x00type_sizesq\x03}q\x04(X\x05\x00\x00\x00shortq\x05K\x02X\x03\x00\x00\x00intq\x06K\x04X\x04\x00\x00\x00longq\x07K\x04uu.\x80\x02(X\x07\x00\x00\x00storageq\x00ctorch\nFloatStorage\nq\x01X\x0f\x00\x00\x00140190109010112q\x02X\x03\x00\x00\x00cpuq\x03K\nNtq\x04Q.\x80\x02]q\x00X\x0f\x00\x00\x00140190109010112q\x01a.\n\x00\x00\x00\x00\x00\x00\x00\xd9\x07\x99\xbe\xb8|\xd4\xbf~\x997?\xb37\xc5\xbfqRa\xbe\x91\xc3\x02\xbf\xe7\xf7??\xf0G\x1a?@\x99\xa2?K\x92|?\x94\x85\x94R\x94K\x00K\n\x85\x94K\x01\x85\x94\x89\x8c\x0bcollections\x94\x8c\x0bOrderedDict\x94\x93\x94)R\x94t\x94R\x94.',
  'encoder': 'my-enc'}}
```

This dictionary is now in a form compatible with storage in MongoDB or similar document stores. The original `Encoder` is able to decode the data in the opposite direction:

```python
>>> enc.decode(e.encode()['_content']['bytes'])
```

SuperDuperDB uses the `"_content.encoder"` subfield to determine which `Encoder` instance it should use.
`Encoder` instances are loaded from SuperDuperDB using the [inbuilt serialization mechanism](serialization).

The `Encoder` may also be used to add content via a reference given in the `uri` parameter. For example:

```python
>>> enc(uri='file://data/images/my_tensor.pkl').encode()
{'_content': {'uri': 'file://data/img/tensor.pkl', 'encoder': 'my-enc'}}
```

This is also a document which can be handled by the database. SuperDuperDB then creates a job which pulls in content mentioned in the `uri` subfield.