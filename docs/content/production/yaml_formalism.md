# YAML/ JSON formalism

`superduperdb` includes a mapping back and forth from Python to YAML/ JSON formats.
The mapping is fairly self-explanatory after reading the example below.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


<Tabs>
    <TabItem value="YAML" label="YAML" default>

        ```yaml
        _base: "?my_vector_index"
        _leaves:
          postprocess:
            _path: superduperdb/base/code/Code
            code: '
              from superduperdb import code

              @code
              def postprocess(x):
                  return x.tolist()
              '
          my_vector:
            _path: superduperdb/components/vector_index/vector
            shape: 384
          sentence_transformer:
            _path: superduperdb/ext/sentence_transformers/model/SentenceTransformer
            datatype: "?my_vector"
            model: "all-MiniLM-L6-v2"
            postprocess: "?postprocess"
          my_query:
            _path: superduperdb/backends/mongodb/query/parse_query
            query: "documents.find()"
          my_listener:
            _path: superduperdb/components/listener/Listener
            model: "?sentence_transformer"
            select: "?my_query"
            key: "X"
          my_vector_index:
            _path: superduperdb/components/vector_index/VectorIndex
            indexing_listener: "?my_listener"
            measure: cosine
        ```

        Then from the commmand line:

        ```bash
        superduperdb apply --manifest='<path_to_config>.yaml'
        ```

    </TabItem>
    <TabItem value="JSON" label="JSON" default>

        ```json
        {
          "_base": "?my_vector_index",
          "_leaves": {
            "postprocess": {
              "_path": "superduperdb/base/code/Code",
              "code": "from superduperdb import code\n\n@code\ndef postprocess(x):\n    return x.tolist()"
            },
            "my_vector": {
              "_path": "superduperdb/components/vector_index/vector",
              "shape": 384
            },
            "sentence_transformer": {
              "_path": "superduperdb/ext/sentence_transformers/model/SentenceTransformer",
              "datatype": "?my_vector",
              "model": "all-MiniLM-L6-v2",
              "postprocess": "?postprocess"
            },
            "my_query": {
              "_path": "superduperdb/backends/mongodb/query/parse_query",
              "query": "documents.find()"
            },
            "my_listener": {
              "_path": "superduperdb/components/listener/Listener",
              "model": "?sentence_transformer",
              "select": "?my_query"
            },
            "my_vector_index": {
              "_path": "superduperdb/components/vector_index/VectorIndex",
              "indexing_listener": "?my_listener",
              "measure": "cosine"
            }
          }
        }
        ```

        Then from the command line:

        ```bash
        superduperdb apply --manifest='<path_to_config>.json'
        ```

    </TabItem>
    <TabItem value="Python" label="Python" default>

        ```python
        from superduperdb import superduper
        from superduperdb.components.vector_index import vector
        from superduperdb.ext.sentence_transformers.model import SentenceTransformer
        from superduperdb.components.listener import Listener
        from superduperdb.components.vector_index import VectorIndex
        from superduperdb.base.code import Code
        from superduperdb import Stack


        db = superduper('mongomock://')

        datatype = vector(shape=384, identifier="my-vec")


        def postprocess(x):
            return x.tolist()


        postprocess = Code.from_object(postprocess)


        model = SentenceTransformer(
            identifier="test",
            datatype=datatype,
            predict_kwargs={"show_progress_bar": True},
            signature="*args,**kwargs",
            model="all-MiniLM-L6-v2",
            device="cpu",
            postprocess=postprocess,
        )

        listener = Listener(
            identifier="my-listener",
            key="txt",
            model=model,
            select=db['documents'].find(),
            active=True,
            predict_kwargs={}
        )

        vector_index = VectorIndex(
            identifier="my-index",
            indexing_listener=listener,
            measure="cosine"
        )

        db.apply(vector_index)
        ```
      
    </TabItem>
</Tabs>