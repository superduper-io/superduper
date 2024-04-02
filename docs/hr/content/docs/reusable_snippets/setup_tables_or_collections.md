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
        from superduperdb import Schema
        
        schema = None
        if isinstance(datatype, DataType):
            schema = Schema(fields={'x': datatype})
            db.add(schema)        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.backends.ibis import Table
        from superduperdb.backends.ibis.field_types import FieldType
        
        if isinstance(datatype, DataType):
            schema = Schema(fields={'x': datatype})
        else:
            schema = Schema(fields={'x': FieldType(datatype)})
        db.add(Table('documents', schema=schema))        
        ```
    </TabItem>
</Tabs>
