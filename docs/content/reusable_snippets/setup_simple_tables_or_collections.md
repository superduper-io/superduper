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
        # If our data is in a format natively supported by MongoDB, we don't need to do anything.
        from superduperdb.backends.mongodb import Collection
        
        table_or_collection = Collection('documents')
        select = table_or_collection.find({})        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.backends.ibis import Table
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
        
        select = table_or_collection.select("id", "prompt", "response")        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="setup_simple_tables_or_collections.md" />
