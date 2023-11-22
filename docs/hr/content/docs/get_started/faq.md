---
sidebar_position: 5
---

# FAQ
Learn more about SuperDuperDB.

<details>
<summary>What's the difference between SuperDuperDB and MindsDB?</summary>

The main differences between SuperDuperDB and MindsDB are outlined below:

1. **Developer Experience:**
   - **MindsDB:** Requires learning a new language created by MindsDB.
   - **SuperDuperDB:** Only requires proficiency in Python and familiar database query languages/operators.

2. **Integration with Python Ecosystem:**
   - **MindsDB:** Utilizes a cloaked connection between data sources and Python models.
   - **SuperDuperDB:** Offers full transparency in the notebook, allowing for granular output inspection, debugging, and visualization within known Python environments and tools.

3. **Vector Search:**
   - **MindsDB:** Does not support native vector embedding, resulting in no vector search capability.
   - **SuperDuperDB:** Supports vectors in any format, including raw, and integrates various vector search solutions.

4. **Support for Flexible Data Types:**
   - **MindsDB:** Limited to handling text and numbers only.
   - **SuperDuperDB:** Supports any datatype, providing flexibility in managing diverse data types.

5. **Multi-Tenant (Multi Data Store):**
   - **MindsDB:** Does not separate data source, model registry, and metadata.
   - **SuperDuperDB:** Allows different locations for data stores, models, and metadata, facilitating multi-data store setups that can share the same models.

6. **Bringing Your Own Models:**
   - **MindsDB:** Requires rewriting and reorganization of your model when bringing your own models.
   - **SuperDuperDB:** Enables the use of your framework natively without requiring any adaptation when bringing your own models.
</details>

<details>
<summary>Is SuperDuperDB a database?</summary>

No, SuperDuperDB is not a traditional standalone database. Instead, it is a versatile Python framework or tool designed to seamlessly integrate artificial intelligence capabilities into various databases. It supports a wide range of databases, including but not limited to MongoDB, MySQL, Postgres, and more. The focus is on enhancing database functionality with AI features rather than serving as a standalone database solution.
</details>

<details>
<summary>Is SuperDuperDB a vector-database?</summary>

No, SuperDuperDB is not a vector-database. It is a versatile Python framework that excels in bringing AI into your favorite database.
</details>
 