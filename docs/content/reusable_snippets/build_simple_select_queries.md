---
sidebar_label: Build simple select queries
filename: build_simple_select_queries.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Build simple select queries


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        
        select = table_or_collection.find({})        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        
        select = table_or_collection.to_query()        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="build_simple_select_queries.md" />
