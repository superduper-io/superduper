---
sidebar_label: Start your system
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Start your system


<Tabs>
    <TabItem value="Development" label="Development" default>
        ```python
        # Nothing to do here (everything runs in-process)        
        ```
    </TabItem>
    <TabItem value="Experimental Cluster" label="Experimental Cluster" default>
        ```python
        !python -m superduperdb local_cluster        
        ```
    </TabItem>
    <TabItem value="Docker-Compose" label="Docker-Compose" default>
        ```python
        !make testenv_image
        !make testenv_init        
        ```
    </TabItem>
</Tabs>
