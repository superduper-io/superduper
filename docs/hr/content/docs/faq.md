---
sidebar_position: 5
---

# FAQ
Learn more about SuperDuperDB.

<details>
<summary>Is SuperDuperDB a database?</summary>

No, SuperDuperDB is not a traditional standalone database. Instead, it is a versatile Python framework or tool designed to seamlessly integrate artificial intelligence capabilities into various databases. It supports a wide range of databases, including but not limited to MongoDB, MySQL, Postgres, and more. The focus is on enhancing database functionality with AI features rather than serving as a standalone database solution.
</details>

<details>
<summary>Is SuperDuperDB a vector-database?</summary>

No, SuperDuperDB is not a vector-database. It is a versatile Python framework that excels in bringing AI into your favorite database.
</details>

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
   - **MindsDB:** Does not support native vector embedding, resulting in no natively supported vector search capability.
   - **SuperDuperDB:** Supports vectors in any format, including raw, and integrates various vector search solutions; this includes images and videos.

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
<summary>Is plugging `superduperdb` directly into a database secure? What precautions are in place, and can I restrict access to specific tables, such as a users table?</summary>

To adhere to the principle of least privilege, SuperDuperDB requires read-only access to the tables you intend to `index`.

One option is maintaining your database as read-only and storing the index externally, such as on your filesystem. Alternatively, you can establish a new table dedicated to housing the index (e.g superduper_index). In this case, the requisite step would be granting us write access to that specific table.

For enhanced security, consider creating a new user specifically for SuperDuperDB. Grant this user read-only access to your data tables and read-write access exclusively to the `superduper_index` table.

If you value privacy as well, we recommend engaging in a more in-depth discussion within the project's Slack channel: [SuperDuperDB Slack](https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA).

</details>
 