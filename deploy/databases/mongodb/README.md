# MongoDB Server


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

Mongo requires some initialization steps. For that, we provide a `wait_ready.sh` script for waiting
until the Mongo server is ready.