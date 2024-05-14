# PostgreSQL Server


## Server Installation Instructions

To begin using the database server, execute the following command:
 ```
 make init-db
 ```

To halt the database server, use the following command:
 ```bash
 make shutdown-db
 ```

## Client Execution Guidelines

The `example.py` serves as a basic client script for establishing a connection with the database. 

As certain databases may necessitate additional libraries, it is imperative to install these prerequisites before utilizing the client.

Execute the subsequent command to install the required dependencies:

```bash
make requirements
```

Subsequently, run the client by executing:


```bash
make run-example
```



# Known Issues

**Question:** I'm getting the following issue: `Error initializing to DataBackend Client: libodbc.
so.2: cannot open shared object file: No such file or directory`

**Answer:** You need to install `unixODBC` package on your system.