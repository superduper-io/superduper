---
sidebar_label: Start your cluster
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Start your cluster

:::note
Starting a SuperDuperDB cluster is useful in production and model development
if you want to enable scalable compute, access to the models by multiple users for collaboration, 
monitoring.

If you don't need this, then it is simpler to start in development mode.
:::


<Tabs>
    <TabItem value="Experimental Cluster" label="Experimental Cluster" default>
        ```python
        !python -m superduperdb local-cluster up        
        ```
    </TabItem>
    <TabItem value="Docker-Compose" label="Docker-Compose" default>
        ```python
        !make testenv_image
        !make testenv_init        
        ```
    </TabItem>
</Tabs>
```python
from superduperdb import superduper

db = superduper()
```

