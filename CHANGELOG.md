# superduper.io Changelog

All notable changes to this project will be documented in this file.

The format is inspired by (but not strictly follows) [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Before you create a Pull Request, remember to update the Changelog with your changes.**

## Changes Since Last Release

#### Changed defaults / behaviours

- Change images docker superduper/<image> to superduperio/<image> 
- Change the image's user from `/home/superduperdb` to `/home/superduper`
- Add message broker service config
- Add helper dict method in Event.
- Use declare_component from base class.
- Use different colors to distinguish logs
- Change all the `_outputs.` to `_outputs__`
- Disable cdc on output tables.
- Remove `-` from the uuid of the component.
- Add _execute_select and filter in the Query class.
- Optimize the logic of ready_ids in trigger_ids.
- Move all plugins superduperdb/ext/* to /plugins
- Optimize the logic for file saving and retrieval in the artifact_store.
- Add backfill on load of vector index
- Add startup event to initialize db.apply jobs
- Update job_id after job submission
- Fixed default event.uuid
- Fixed atlas vector search
- Fix the bug where shared artifacts are deleted when removing a component.
- Fix compatibility issues with the latest version of pymongo.
- Fix the query parser incompatibility with '.' symbol.
- Fix the post like in the service vector_search.
- Fix the conflict of the same identifier during encoding.
- Fix the issue where MongoDB did not create the output table.
- Fix the bug in the CI where plugins are skipping tests.
- Updated CONTRIBUTING.md
- Add README.md files for the plugins.
- Add templates to project 
- Add frontend to project 
- Change compute init order in cluster initialize
- Add table error exception and sql table length fallback.
- Permissions of artifacts increased
- Make JSON-able a configuration depending on the databackend
- Restore some training test cases
- Simple querying shell
- Fix existing templates
- Add optional data insert to `Table`

#### New Features & Functionality

- Modify the field name output to _outputs.predict_id in the model results of Ibis.
- Remove the document_embed function.
- Support MongoDB outputs query
- Make "create a table" compulsory
- All datatypes should be wrapped with a Schema
- Support eager mode
- Add CSN env var
- Make tests configurable against backend
- Make the prefix of the output string configurable
- Add testing utils for plugins
- Add `cache` field in Component
- Add predict_id in Listener
- Add serve in Model
- Added templates directory with OSS templates
- Qdrant vector search support
- Add placeholder for web app link in Application
- Add support for remote artifacts
- Add basic rest server
- Add `@trigger` decorator to improve developer experience
- `ModelRouter` to enable easy toggles
- Simple interactive shell
- Add pdf rag template
- Add sql table length exceed limit and uuid truncation.

#### Bug Fixes

- Vector-search vector-loading bug fixed
- Bugs related to new queuing paradigm
- Remove --user from make install_devkit as it supposed to run on a virtualenv.
- component info support list
- Trigger downstream vector indices.
- Fix vector_index function job.
- Fix verbosity in component info
- Change default encoding to sqlvector
- Fix some links in documentation 
- Change `__dataclass_params__` to `_dataclass_params`
- Make component reload after caching in apply
- Fix a minor bug in schedule_jobs
- Fix vector index cleanup
- Fix the condition of the CDC job.
- Fix form_template
- Fix the duplicate children in the component.
- Fix datatype for graph models
- Fix bug in variables
- Fix Qdrant collection name
- Fix the ordering and sequencing of jobs initiated on `db.apply`
- Fix rest routes with db injection and prefix.
- Fix cluster db setter.

## [0.3.0](https://github.com/superduper-io/superduper/compare/0.3.0...0.2.0])    (2024-Jun-21)

#### Changed defaults / behaviours

- Renamed superduper -> superduper
- Added data_pipeline_deps test case

#### New Features & Functionality

- Add plugin component.
- QueryTemplate component
- Support for packaging application from the database.
- Added DataInit component
- Refactor ray jobs

#### Bug Fixes

- Fix templates
- Fix the issue of the filter in select not working in the listener.
- Fix exports
- Fix model query
- Fix doc-strings
- Fix support for keys in new queue handler.
- Fix the bug where the query itself changes after encoding
- Fix the dependency error in copy_vectors within vector_index. 
- Fix Template substitutions
- Fix remove un_use _import function
- Fix some linting and small refactors.
## [0.2.0](https://github.com/superduper-io/superduper/compare/0.1.3...0.2.0])    (2024-Jun-21)

#### Changed defaults / behaviours

- Run Tests from within container
- Add model dict output indexing in graph
- Make lance upsert for added vectors
- Make vectors normalized in inmemory vector database for cosine measure
- Add local cluster as tmux session 
- At the end of the test, drop the collection instead of the database
- Force load vector indices during backfill
- Fix pandas database (in-memory)
- Add and update docstrings in component classes and methods
- Changed the rest implementation to use new serialization
- Remove unused deadcode from the project
- Auto wrap insert documents as Document instances
- Changed the rest implementation to use the new serialization
- Mask special character keys in mongodb queries
- Fix listener cleanup after removal
- Don't require `@dc.dataclass` or `@merge_docstrings` decorator
- Make output of `Document.encode()` more minimalistic
- Increment minimum supported ibis version to 9.0.0
- Make database connections reconnection on token expiry
- Prototype the cron job services
- Simplified Taskworkflow

#### New Features & Functionality

- Add nightly image for pre-release testing in the cloud environment
- Fix  torch model fit and make schedule_jobs at db add
- Add requires functionality for all extension modules
- CI fails if CHANGELOG.md is not updated on PRs
- Update Menu structure and renamed use-cases
- Change and simplify the contract for writing new `_Predictor` descendants (`.predict_one`, `.predict`)
- Add file datatype type to support saving and reading files/folders in artifact_store
- Create models directly by importing package from auto and with decorator `@objectmodel`, `@torchmodel`
- Support Schema option for MongoDB
- Optimize LLM fine-tuning
- Sort out the llm directory structure
- Add cache support in inmemory vector searcher
- Add compute_kwargs option for model
- Add BulkWrite mongodb query
- Rename `_Predictor` to `Model`
- Allow developers to write `Listeners` and `Graph` in a single formalism
- Change unittesting framework to pure configuration (no patching configs)
- Add a simple REST server implementation
- Add reusable snippets that are reused across the docs
- Added snippet for connecting to superduper in docs
- Added support to serialize documents in a flat way "_leaves"
- Added `lazy_file` datatype
- Show the DataLayer configuration
- Optimized LLM finetuning usage experience
- Auto-infer Schema from data
- Lazy-creation of output tables for ibis to enable auto-inference of output schema
- Add database packages that improve deployment and connection testing
- Enable dependency injection on image builders
- Add database package for oracle
- Reconstruct data serialization and database queries
- Auto-create tables and schemas
- Add `Application` and `Template` support to build reusable apps
- Add pretty-print to `Component.info`
- `Model` 
- 'Add pluggable compute backend via config'

#### Bug Fixes

- Fixed cross platfrom issue in cli command
- Separate nightly release from sandbox
- Fixed a bug in refresh_after_insert for listeners with select None
- Refactor graph internal with input mapping
- Fixed a bug in Component init
- Fixed a bug in predict in db for missing ouptuts
- Fixed a bug in variable set
- Fixed the bug where select in listener is modified in schedule_jobs.
- LLM CI random errors
- VectorIndex schedule_jobs missing function
- Fixed some bugs of the cdc RAG application
- Fixed open source RAG Pipeline
- Fixed vllm real-time task concurrency bug
- Fixed Post-Like feature
- Added CORS Policy regarding REST server implementation
- Fixed some bugs in multimodal usecase
- Fixed File datatype
- Fixed a bug in artifact store to skip duplicate artifacts
- Fixed database permission issues when connecting to mongodb
- Handle ProgrammingError of SnowFlake for non-existing objects
- Updated the use cases.
- Update references to components and artifacts.
- Fix Ray compute async with job submission api. 
- Refactor document encode
- Change '_leaves' to '_builds'
- Fixed empty identifier of Code.from_object.
- Fixed Native encodable.
- Fix ibis cdc and cdc config
- Fixed 'objectmodel' and 'predict_one' in doc.
- Fixed ray dependencies bug.
- Fixed listener dependencies bug.
- Fixed cluster bug.

## [0.1.3](https://github.com/superduper-io/superduper/compare/0.1.1...0.1.3])    (2024-Jun-20)

Test release before v0.2 

## [0.1.1](https://github.com/superduper-io/superduper/compare/0.1.0...0.1.1])    (2024-Feb-09)

#### Changed defaults / behaviours

- Test suite takes config from external .env file
- Added support for multi key in model predict
- Support 3.10+ due to `dataclass` supported features
- Updated the table creation method in MetaDataStore to improve compatibility across various databases
- Replaced JSON data with String format before storage in SQLAlchemy
- Implemented storage of byte data in base64 format
- Migrated MongoDB Atlas vector search as a standalone searcher like lance
- Deprecated Demo Image. Now Notebooks run in Colab
- Replace dask with ray compute backend
- All training and validation parameters to be configured in `_Predictor` attributes (`.trainer`, `.train_X`, etc.)
- Docker build can include optional custom `requirements.txt` path

#### New Features & Functionality

- Add Llama cpp model in extensions.
- Basic Ray server support to server models on ray cluster
- Add Graph mode support to chain models
- Simplify the testing of SQL databases using containerized databases
- Integrate Monitoring(cadvisor/Prometheus) and Logging (promtail/Loki) with Grafana, in the `testenv`
- Add `QueryModel` and `SequentialModel` to make chaining searches and models easier
- Add `insert_to=<table-or-collection>` to `.predict` to allow single predictions to be saved.
- Support vLLM (running locally or remotely on a ray cluster)
- Support LLM service in OpenAI format
- Add lazy loading of artifacts by default

#### Bug Fixes

- Update connection uris in `sql_examples.ipynb` to include snippets for Embedded, Cloud, and Distributed databases
- Fixed a bug related to using Clickhouse as both databackend and metastore

## [0.1.0](https://github.com/superduper-io/superduper/compare/0.0.20...0.1.0])    (2023-Dec-05)

#### New Features & Functionality

- Introduced Chinese version of README

#### Bug Fixes

- Updated paths for docker-compose.

## [0.0.20](https://github.com/superduper-io/superduper/compare/0.0.10...0.0.20])    (2023-Dec-04)

#### Changed defaults / behaviours

- Chop down large files from the history to reduce the size of the repo

## [0.0.19](https://github.com/superduper-io/superduper/compare/0.0.15...0.0.19])    (2023-Dec-04)  

#### Changed defaults / behaviours

- Add Changelog for tracking changes on the repo. It must be filled before any PR
- Remove ci-pinned-dependencies and replaced them with actions with better cache management
- Change logging mechanism from the default to loguru
- Update icons on the README.
- Reboot test-suite, with modular approach to toggling between SQL and MongoDB tests
- Add model-versioning of model-outputs
- Refactor OpenAI code to use the new features of the OpenAI API
- Fixes for dask worker compute delegation
- Wrap compute with abstraction as component of datalayer
- Simplify approach to project configuration
- Add services for vector-search and CDC for more comprehensive cluster mode
- Add a `Component.post_create` hook to enable logic to incorporate model versions
- Fix multiple issues with `ibis`/ SQL code

#### New Features & Functionality

- Add support for selecting whether logs will be redirected to the system output or directly to Loki

#### Bug Fixes

- Added libgl libraries in Dockerfile to correctly render the video in notebooks

## [0.0.15](https://github.com/superduper-io/superduper/compare/0.0.14...0.0.15])    (2023-Nov-01)

#### Changed defaults / behaviors

-   Updated readme by @fnikolai in #1196.
-   Removed unused import by @jieguangzhou in #1205.
-   Updated README.md with contributors by @thejumpman2323 in #1201.
-   Added conditional builders in Dockerfile by @fnikolai in #1213.
-   Optimized unit tests by @jieguangzhou in #1204.

#### New Features & Functionality

-   Updated README.md with announcement emoji by @thejumpman2323 in #1222.
-   Launched announcement by @fnikolai in #1208.
-   Added raw SQL in ibis by @thejumpman2323 in #1220.
-   Added experimental keyword by @fnikolai in #1218.
-   Added query table by @thejumpman2323 in #1212.
-   Merged Ashishpatel26 main by @blythed in #1224.
-   Bumped Version to 0.0.15 by @fnikolai in #1225.

#### Bug Fixes

-   Fixed dependencies and makefile by @fnikolai in #1209.
-   Fixed demo release by @fnikolai in #1210.

## [0.0.14](https://github.com/superduper-io/superduper/compare/0.0.13...0.0.14])    (2023-Oct-27)

## [0.0.13](https://github.com/superduper-io/superduper/compare/0.0.12...0.0.13])    (2023-Oct-19)

## [0.0.12](https://github.com/superduper-io/superduper/compare/0.0.11...0.0.12])    (2023-Oct-12)

## [0.0.11](https://github.com/superduper-io/superduper/compare/0.0.10...0.0.11])    (2023-Oct-10)

## [0.0.10](https://github.com/superduper-io/superduper/compare/0.0.9...0.0.10])    (2023-Oct-09)

## [0.0.9](https://github.com/superduper-io/superduper/compare/0.0.8...0.0.9])      (2023-Oct-06)

## [0.0.8](https://github.com/superduper-io/superduper/compare/0.0.7...0.0.8])      (2023-Sep-29)

## [0.0.7](https://github.com/superduper-io/superduper/compare/0.0.6...0.0.7])      (2023-Sep-14)

## [0.0.6](https://github.com/superduper-io/superduper/compare/0.0.5...0.0.6])      (2023-Aug-29)

## [0.0.5](https://github.com/superduper-io/superduper/compare/0.0.5...0.0.4])      (2023-Aug-15)

## 0.0.4      (2023-Aug-03)


