---
sidebar_label: Setup tables or collections
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Setup tables or collections


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        # Note this is an optional step for MongoDB
        # Users can also work directly with `DataType` if they want to add
        # custom data
        from superduperdb import Schema, DataType
        from superduperdb.backends.mongodb import Collection
        
        table_or_collection = Collection('documents')
        USE_SCHEMA = False
        
        if USE_SCHEMA and isinstance(datatype, DataType):
            schema = Schema(fields={'x': datatype})
            db.apply(schema)        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.backends.ibis import Table
        from superduperdb import Schema, DataType
        from superduperdb.backends.ibis.field_types import dtype
        
        datatype = "str"
        
        if isinstance(datatype, DataType):
            schema = Schema(identifier="schema", fields={"id": dtype("str"), "x": datatype})
        else:
            schema = Schema(
                identifier="schema", fields={"id": dtype("str"), "x": dtype(datatype)}
            )
        
        table_or_collection = Table('documents', schema=schema)
        
        db.apply(table_or_collection)        
        ```
    </TabItem>
</Tabs>
