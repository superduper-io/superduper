# Architecture

Here is a schematic of the `superduperdb` design.

![](/img/light.png)

### Explanation

1. `superduperdb` expects data and components to be added/ updated from a range of client-side mechanisms: **scripts**, **apps**, **notebooks** or **third-party database clients** (possibly non-python).

1. Users and programs can add **components** (**models**, data **encoders**, **vector-indexes** and more) from the client-side. These large items are stored in the **artifact-store** and are tracked via the **meta-data** store.

1. If data is inserted to the **databackend** the **change-data-capture (CDC)** component captures these changes as they stream in.

1. **(CDC)** triggers **work** to be performed in response to these changes, depending on which **components** are present in the system.

1. The **work** is submitted to the **workers** via the **scheduler**. Together the **scheduler** and **workers** make up the **compute** layer.

1. **workers** write their outputs back to the **databackend** and trained models to the **artifact-store**

1. The **compute**, **databackend**, **metadata-store**, **artifact-store** collectively make up the **datalayer**

1. The **datalayer** may be queried from client-side, including hybrid-queries or **compound-select** queries, which synthesizes classical **selects** with **vector-searches**