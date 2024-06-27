---
sidebar_label: Setup simple tables or collections
filename: setup_simple_tables_or_collections.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Setup simple tables or collections


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        # If data is in a format natively supported by MongoDB, we don't need to do anything.
        # However to manually specify datatypes, do as below
        from superduperdb import Schema
        from superduperdb.ext.pillow import pil_image
        from superduperdb.components.datatype import pickle_serializer
        
        fields = {
            'serialized_content': pickle_serializer,
            'img_content': pil_image_hybrid,
        }
        
        schema = Schema(identifier="my-schema", fields=fields)
        db.apply(schema)

        # Now assert `Document` instances, specifying this schema
        db['documents'].insert_many([
            Document({
                'serialized_content': item,
                'img_content': img,
            }, schema='my-schema')
            for item, img in data
        ])
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb import Table
        from superduperdb import Schema, DataType
        from superduperdb.backends.ibis.field_types import dtype
        
        for index, data in enumerate(datas):
            data["id"] = str(index) 
        
        fields = {}
        
        for key, value in data.items():
            fields[key] = dtype(type(value))
        
        schema = Schema(identifier="schema", fields=fields)
        
        table_or_collection = Table('documents', schema=schema)
        
        db.apply(table_or_collection)

        db['documents'].insert([
            {'prompt', prompt, 'response': response}
            for prompt, response in data
        ])
        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="setup_simple_tables_or_collections.md" />
