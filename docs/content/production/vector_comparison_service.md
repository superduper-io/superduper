# Vector-searcher service

The vector-comparison service is a standalone, 
`(id, vector)` only, vector-database, which may be 
deployed to externalize vector-search from the databackend.

Here's how to deploy it:

```python
superduper vector-searcher
```

Here are the endpoints:

### Create searcher

  - **Method**: `POST`
  - **Endpoint**: `/create`
  - **Parameters**:
    
    | name | type | description | required |
    | --- | --- | --- | --- |
    | `vector_index` | string | Name of the corresponding `VectorIndex` | yes |
    | `measure` | string | Type of measure function to compare vectors | yes |
  

### List searchers

  - **Method**: `GET`
  - **Endpoint**: `/list`


### Add vectors to searcher

  - **Method**: `POST`
  - **Endpoint**: `/add`
  - **Parameters**:
    
    | name | type | description | required |
    | --- | --- | --- | --- |
    | `vector_index` | string | Name of the corresponding `VectorIndex` | yes |
    | `vectors` | JSON | list of `(id, vector)` | yes |


### Remove vectors from searcher

  - **Method**: `POST`
  - **Endpoint**: `/remove`
  - **Parameters**:
    
    | name | type | description | required |
    | --- | --- | --- | --- |
    | `vector_index` | string | Name of the corresponding `VectorIndex` | yes |
    | `vectors` | JSON | list of `id` | yes |


### Delete searcher

  - **Method**: `POST`
  - **Endpoint**: `/remove`
  - **Parameters**:
    
    | name | type | description | required |
    | --- | --- | --- | --- |
    | `vector_index` | string | Name of the corresponding `VectorIndex` | yes |

