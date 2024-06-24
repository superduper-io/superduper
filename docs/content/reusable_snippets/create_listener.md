---
sidebar_label: Create Listener
filename: create_listener.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Create Listener

## Two ways to define listener


<Tabs>
    <TabItem value="Listener" label="Listener" default>
        ```python
        from superduperdb import Listener
        db.apply(
            Listener(
                key='key_name',
                model=model,
                select=select,
            )
        )        
        ```
    </TabItem>
    <TabItem value="model.to_linstener" label="model.to_linstener" default>
        ```python
        db.apply(model.to_listener(key='key_name', select=select))        
        ```
    </TabItem>
</Tabs>
## Data passed into the model


<Tabs>
    <TabItem value="Single Field" label="Single Field" default>
        ```python
        # Model predict function definition: model.predict(x)
        # Data example in database: {"key_name": 10}
        # Then the listener will call model.predict(10)
        from superduperdb import Listener
        db.apply(
            Listener(
                key='key_name',
                model=model,
                select=select,
            )
        )        
        ```
    </TabItem>
    <TabItem value="Multiple fields(*args)" label="Multiple fields(*args)" default>
        ```python
        # Model predict function definition: model.predict(x1, x2)
        # Data example in database: {"key_name_1": 10, "key_name_2": 100}
        # Then the listener will call model.predict(10, 100)
        from superduperdb import Listener
        db.apply(
            Listener(
                key=['key_name_1', 'key_name_2'],
                model=model,
                select=select,
            )
        )        
        ```
    </TabItem>
    <TabItem value="Multiple fields(*kwargs)" label="Multiple fields(*kwargs)" default>
        ```python
        # Model predict function definition: model.predict(x1, x2)
        # Data example in database: {"key_name_1": 10, "key_name_2": 100}
        # Then the listener will call model.predict(x1=10, x2=100)
        from superduperdb import Listener
        db.apply(
            Listener(
                key={"key_name_1": "x1", "key_name_2": "x2"},
                model=model,
                select=select,
            )
        )        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="create_listener.md" />
