# Super Duper DB

## Installation

### Mac OSX

Install redis and mongodb

```
brew install redis
brew install mongodb
```

Then install the python requirements

```
pip install -r requirements.txt
```

## Architecture

![](./img/architecture.png)

1. Client - run on client to send off requests to various work horses.
1. MongoDB - standard mongo deployment.
1. Vector lookup - deployment of sddb with faiss/ scaNN.
1. Job master - master node for job cluster, sends messages to redis (rq)
1. Redis database - database instance for rq master
1. Job worker - worker node(s) for jobs, computes vectors, and performs model trainings.
   Retrieves jobs from redis.
   
## TODO

- concept of multiple downloads per collection (e.g. download website, apply extractor, get urls)
- model registry in mongo
- some models allowed to go into main record and not into `_outputs`
- models in mongodb filesystem https://pymongo.readthedocs.io/en/stable/examples/gridfs.html
