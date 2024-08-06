# Developer vs. production mode

Please refer to the [architecture](../fundamentals/design.md) page for a detailed description of the `superduper` architecture.

There are several important services in a `superduper` setup which may be run in-process, or in their 
own micro-services and containers:

- `jupyter` notebook/ client
- change-data-capture (CDC) service
- compute
  - scheduler
  - workers
- vector-searcher service
- REST-ful server service

## Development mode

With the default settings of `superduper`, all of these items run in a single process.
This is great for:

- Debugging
- Prototyping
- Experimentation
- Exploration
- Querying `superduper`

## Production

There are several gradations of a more productionized deployment.
In the most distributed case we have:

- A `jupyter` environment running in its own process
- A [distributed **Ray** cluster](non_blocking_ray_jobs), with scheduler and workers configured to work with `superduper`
- A [**change-data-capture** service](change_data_capture)
- A [**vector-search** service](vector_comparison_service), which finds similar vectors, given an input vector
- A [**REST** server](./rest_api)

In the remainder of this section we describe the use of each of these services
