**************************
SuperDuperDB Documentation
**************************

SuperDuperDB is:

* An AI-database management system
    The AI data/ model workflow is highly complex with multiple interdependent steps where
    models and data repeatedly interface with one another.
    SuperDuperDB aims to be exactly the right environment to bring the data and AI worlds together.
* Shares the standard advantages of a database
    * Access control
    * User groups
    * Security
    * Scalability
    * Indexing
* Adds novel features to databases which are not possible without AI
    * Arbitrary data imputations
    * Semantic indexing based on linear algebra
    * Associating AI models directly with the data which trained them for transparent model lineage.
    * Allowing AI models to reference items from the database in flexible ways (e.g. the most similar items to the current data point)
* Integrates PyTorch models natively to the database
    Allowing users to work intuitively and efficiently with AI.
* Is ready for the latest generation of AI models
    New developments in AI imply that architectures such as `Retro <https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens>`_
    will need ready access to your data. SuperDuperDB allows users to build models which reference
    the database explicitly.
* Is fully modular and allows users to
    * provide arbitrary PyTorch models
    * apply arbitrary pre- and post-processing
    * define custom data types, e.g. tensors, images, audio, video
* Enables training models directly on the database
    SuperDuperDB has prebaked support for training semantic lookup tasks and imputation style
    learning tasks. User-defined training setups may also be
    supplied and stored in the database. The technology leverages data loaders which enable direct
    cross talk between MongoDB and PyTorch.

Documentation contents
======================

.. toctree::
    :maxdepth: 1

    getting_started
    overview
    concepts
    high_level_usage
    full_usage

