# Production features in `superduperdb`

`superduperdb` was made with productionization in mind. These means several things:

## Modifiable, configurable, flexible

A production ready solution should not come up short, as soon as developers 
encounted a use-case or scenario outside of the norm, or the documented 
applications. In this respect, modifiablility, configurability and flexibility are key.

`superduperdb` contains fully open-source (Apache 2.0, MIT, BSD 3-part) components for all aspects of the setup.
This goes from the AI-models integrated, the databases and client libraries supported, as well as 
the cluster and compute management. This means that developers are never left hung out 
to dry with regard to taking action in altering and interrogating the source doe, as well 
as adding their own functionality.

In addition, `superduperdb` may be used and configured in a wide variety of ways.
It can be used "in-process", with computations blocking ("developer mode") and 
it can be operated in a cluster-like architecture, with compute, vector-search,
change-data capture and a REST-ful server separated into separate services.
This is ideal for teams of developers looking to productionize their AI-data setups.

## Scalability

A production ready solution should scale with the amount of traffic, data
and computation to the system. `superduperdb` includes a `ray` integration
which allows developers to scale the compute as the amount of data and requests
to the system grod. Read more [here](./non_blocking_ray_jobs).

In addition `superduperdb` has the option to separate the vector-comparison and sorting component
into a separate service, so this doesn't block or slow down the main program running.

## Interoperability

Due to the [change-data-capture component](./change_data_capture), developers 
are not required to operate their database through `superduperdb`. Third-party 
database clients, and even other programming languages other than Python 
may be used to add data to the database. Nonetheless, `superduperdb` 
will still process this data.

In addition the [REST API](./rest_api) may be easily used to access `superduperdb`
from the web, or from other programming environments.

## Live serving

The [REST API](./rest_api) service may be used to access `superduperdb` using pure JSON, 
plus references to saved/ uploaded binaries. This gives great flexibility to application
developers looking to build on top of `superduperdb` from Javascript etc..

## SuperDuper protocol

All `superduperdb` components may be built using Python, or directly in a YAML/ JSON formalism
usng the ["superduper-protocol"](./superduper_protocol.md).
This provides a convenient set of choices for AI engineers, infrastructure engineers 
and beyond to share and communicate their AI-data setups in `superduperdb`