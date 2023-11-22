---
sidebar_position: 4
---

# Security

Enhance your understanding of security considerations with SuperDuperDB.

<details>
<summary>Is plugging it directly into a database secure? What precautions are in place, and can I restrict access to specific tables, such as a users table?</summary>

To adhere to the principle of least privilege, SuperDuperDB requires read-only access to the tables you intend to `index`.

One option is maintaining your database as read-only and storing the index externally, such as on your filesystem. Alternatively, you can establish a new table dedicated to housing the index (e.g superduper_index). In this case, the requisite step would be granting us write access to that specific table.

For enhanced security, consider creating a new user specifically for SuperDuperDB. Grant this user read-only access to your data tables and read-write access exclusively to the `superduper_index` table.

If you value privacy as well, we recommend engaging in a more in-depth discussion within the project's Slack channel: [SuperDuperDB Slack](https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA).

</details>