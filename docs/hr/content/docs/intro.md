---
sidebar_position: 1
---

# Welcome to SuperDuperDB!

## What is SuperDuperDB?

SuperDuperDB is an open-source framework for integrating your database with AI models, APIs, and vector search engines, providing streaming inference and scalable training/fine-tuning.

SuperDuperDB is **not** a database. SuperDuperDB is an open platform unifying data infrastructure and AI. Think `db = superduper(db)`: SuperDuperDB transforms your databases into an intelligent system that leverages the full power of the AI, open-source and Python ecosystem. It is a single scalable environment for all your AI that can be deployed anywhere, in the cloud, on-prem, on your machine.

By integrating AI at the data's source, we aim to make building AI applications more easily and flexibly: Not just for use in standard machine learning use-cases like classification, segmentation, recommendation etc., but also generative AI & LLM-chat, vector search as well as highly custom AI applications and workflows involving ultra specialized models.

For more information about SuperDuperDB and why we believe it is much needed, [read this blog post](https://docs.superduperdb.com/blog/superduperdb-the-open-source-framework-for-bringing-ai-to-your-datastore/). 


![](/img/superduperdb.gif)



### Key Features:
- **[Integration of AI with your existing data infrastructure](https://docs.superduperdb.com/docs/docs/walkthrough/apply_models):** Integrate any AI models and APIs with your databases in a single scalable deployment, without the need for additional pre-processing steps, ETL or boilerplate code.
- **[Streaming Inference](https://docs.superduperdb.com/docs/docs/walkthrough/daemonizing_models_with_listeners):** Have your models compute outputs automatically and immediately as new data arrives, keeping your deployment always up-to-date.
- **[Scalable Model Training](https://docs.superduperdb.com/docs/docs/walkthrough/training_models):** Train AI models on large, diverse datasets simply by querying your training data. Ensured optimal performance via in-build computational optimizations.
- **[Model Chaining](https://docs.superduperdb.com/docs/docs/walkthrough/linking_interdependent_models/)**: Easily setup complex workflows by connecting models and APIs to work together in an interdependent and sequential manner.
- **[Simple, but Extendable Interface](https://docs.superduperdb.com/docs/docs/fundamentals/procedural_vs_declarative_api)**: Add and leverage any function, program, script or algorithm from the Python ecosystem to enhance your workflows and applications. Drill down to any layer of implementation, including to the inner workings of your models while operating SuperDuperDB with simple Python commands.
- **[Difficult Data-Types](https://docs.superduperdb.com/docs/docs/walkthrough/encoding_special_data_types/)**: Work directly with images, video, audio in your database, and any type which can be encoded as `bytes` in Python.
- **[Feature Storing](https://docs.superduperdb.com/docs/docs/walkthrough/encoding_special_data_types):** Turn your database into a centralized repository for storing and managing inputs and outputs of AI models of arbitrary data-types, making them available in a structured format and known environment.
- **[Vector Search](https://docs.superduperdb.com/docs/docs/walkthrough/vector_search):** No need to duplicate and migrate your data to additional specialized vector databases - turn your existing battle-tested database into a fully-fledged multi-modal vector-search database, including easy generation of vector embeddings and vector indexes of your data with preferred models and APIs.


**To get started:**
This documentation serves as your comprehensive guide to kickstarting your `superduperdb` journey 🚀

Additionally, you can check the use-cases [here in the docs](https://docs.superduperdb.com/docs/category/use-cases) as well as the apps built by the [superduper community](https://github.com/SuperDuperDB/superduper-community-apps) and try all of them with [live on your browser](https://demo.superduperdb.com/)! 




`Remember, SuperDuperDB is open-source: Please leave a star to support the project! ⭐`
