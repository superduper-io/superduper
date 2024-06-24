---
sidebar_label: Insert data
filename: insert_data.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Insert data

In order to create data, we need to create a `Schema` for encoding our special `Datatype` column(s) in the databackend.


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb import Document, DataType
        
        def do_insert(data, schema = None):
            
            if schema is None and (datatype is None or isinstance(datatype, str)):
                data = [Document({'x': x['x'], 'y': x['y']}) if isinstance(x, dict) and 'x' in x and 'y' in x else Document({'x': x}) for x in data]
                db.execute(table_or_collection.insert_many(data))
            elif schema is None and datatype is not None and isinstance(datatype, DataType):
                data = [Document({'x': datatype(x['x']), 'y': x['y']}) if isinstance(x, dict) and 'x' in x and 'y' in x else Document({'x': datatype(x)}) for x in data]
                db.execute(table_or_collection.insert_many(data))
            else:
                data = [Document({'x': x['x'], 'y': x['y']}) if isinstance(x, dict) and 'x' in x and 'y' in x else Document({'x': x}) for x in data]
                db.execute(table_or_collection.insert_many(data, schema=schema))
        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb import Document
        
        def do_insert(data):
            db.execute(table_or_collection.insert([Document({'id': str(idx), 'x': x['x'], 'y': x['y']}) if isinstance(x, dict) and 'x' in x and 'y' in x else Document({'id': str(idx), 'x': x}) for idx, x in enumerate(data)]))
        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="insert_data.md" />
