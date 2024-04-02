---
sidebar_label: Connect to SuperDuperDB
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Connect to SuperDuperDB


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb import superduper
        
        db = superduper('mongodb://localhost:27017/documents')        
        ```
    </TabItem>
    <TabItem value="MongoMock" label="MongoMock" default>
        ```python
        from superduperdb import superduper
        
        db = superduper('mongomock:///test_db')        
        ```
    </TabItem>
    <TabItem value="SQLite" label="SQLite" default>
        ```python
        from superduperdb import superduper
        
        db = superduper('sqlite://my_db.db')        
        ```
    </TabItem>
    <TabItem value="SQLite-InMemory" label="SQLite-InMemory" default>
        ```python
        from superduperdb import superduper
        
        db = superduper('sqlite://')        
        ```
    </TabItem>
    <TabItem value="PostgreSQL" label="PostgreSQL" default>
        ```python
        from superduperdb import superduper
        
        db = superduper('postgres://localhost:1234')        
        ```
    </TabItem>
</Tabs>
