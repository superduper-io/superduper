# YAML/ JSON formalism

`superduperdb` includes a mapping back and forth to YAML/ JSON formats.
The mapping is fairly self-explanatory after reading the example below.
Note that an important wrinkle, is that the parameters in `dict: ...`
are passed to the imported `cls.handle_integration()` before being
passed onto the `__init__` method of the class. This allows
the class to deal with items which aren't easily expressible directly 
in YAML (JSON) format. For items which are not possible to express
as JSON, the `_BaseEncodable` class and descendants is used.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


<Tabs>
    <TabItem value="YAML" label="YAML" default>

        ```yaml
        identifier: "test"
        _leaves:
          - leaf_type: "component"
            cls: "vector"
            module: "superduperdb.components.vector_index"
            dict:
              shape: 384
              identifier: "my-vec"
        
          - leaf_type: "component"
            cls: "SentenceTransformer"
            module: "superduperdb.ext.sentence_transformers.model"
            dict:
              identifier: "test"
              datatype: "_component/datatype/my-vec"
              predict_kwargs:
                show_progress_bar: true
              signature: "*args,**kwargs"
              model: "all-MiniLM-L6-v2"
              device: "cpu"
              postprocess: |
                from superduperdb import code
                @code
                def my_code(x):
                    return x.tolist()
        
          - leaf_type: "component"
            cls: "Listener"
            module: "superduperdb.components.listener"
            dict:
              identifier: "my-listener"
              key: "txt"
              model: "_component/model/test"
              select:
                documents: []
                query: "documents.find()"
              active: true
              predict_kwargs: {}
        
          - leaf_type: "component"
            cls: "VectorIndex"
            module: "superduperdb.components.vector_index"
            dict:
              identifier: "my-index"
              indexing_listener: "_component/listener/my-listener"
              compatible_listener: null
              measure: "cosine"
        ```

        Then from the commmand line:

        ```bash
        superduperdb apply --manifest='<path_to_yaml>.yml'
        ```

    </TabItem>
    <TabItem value="JSON" label="JSON" default>

        ```json
        {
          "identifier": "test",
          "_leaves": [
            {
              "leaf_type": "component",
              "cls": "vector",
              "module": "superduperdb.components.vector_index",
              "dict": {
                "shape": 384,
                "identifier": "my-vec"
              }
            },
            {
              "leaf_type": "component",
              "cls": "SentenceTransformer",
              "module": "superduperdb.ext.sentence_transformers.model",
              "dict": {
                "identifier": "test",
                "datatype": "_component/datatype/my-vec",
                "predict_kwargs": {
                  "show_progress_bar": true
                },
                "signature": "*args,**kwargs",
                "model": "all-MiniLM-L6-v2",
                "device": "cpu",
                "postprocess": "from superduperdb import code\n\n@code\ndef my_code(x):\n    return x.tolist()\n"
              }
            },
            {
              "leaf_type": "component",
              "cls": "Listener",
              "module": "superduperdb.components.listener",
              "dict": {
                "identifier": "my-listener",
                "key": "txt",
                "model": "_component/model/test",
                "select": {
                  "documents": [],
                  "query": [
                    "documents.find()"
                  ]
                },
                "active": true,
                "predict_kwargs": {}
              }
            },
            {
              "leaf_type": "component",
              "cls": "VectorIndex",
              "module": "superduperdb.components.vector_index",
              "dict": {
                "identifier": "my-index",
                "indexing_listener": "_component/listener/my-listener",
                "compatible_listener": null,
                "measure": "cosine"
              }
            }
          ]
        }
        ```

        Then from the command line:

        ```bash
        superduperdb apply --manifest='<path_to_yaml>.yml'
        ```

    </TabItem>
    <TabItem value="Python" label="Python" default>

        ```python

        from superduperdb import superduper
        from superduperdb.components.vector_index import vector
        from superduperdb.ext.sentence_transformers.model import SentenceTransformer
        from superduperdb.components.listener import Listener
        from superduperdb.components.vector_index import VectorIndex
        from superduperdb import Stack

        from superduperdb.backends.mongodb import Collection

        datatype = vector(shape=384, identifier="my-vec")

        model = SentenceTransformer(
            identifier="test",
            datatype=datatype,
            predict_kwargs={"show_progress_bar": True},
            signature="*args,**kwargs",
            model="all-MiniLM-L6-v2",
            device="cpu",
            postprocess=lambda x: x.tolist(),
        )

        listener = Listener(
            identifier="my-listener",
            key="txt",
            model=model,
            select=Collection('documents').find(),
            active=True,
            predict_kwargs={}
        )

        vector_index = VectorIndex(
            identifier="my-index",
            indexing_listener=listener,
            measure="cosine"
        )

        db = superduper()

        db.apply(
            Stack(
                identifier='test',
                components=[vector_index],
            )
        )
        ```
    </TabItem>
</Tabs>