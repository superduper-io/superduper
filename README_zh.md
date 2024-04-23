<p align="center">
  <a href="https://www.superduperdb.com">
    <img width="90%" src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/img/SuperDuperDB_logo_color.svg">
  </a>
</p>
<div align="center">

# Bring AI to your favorite database! 

</div>

<div align="center">

## <a href="https://superduperdb.github.io/superduperdb/"><strong>Docs</strong></a> | <a href="https://blog.superduperdb.com"><strong>Blog</strong></a> | <a href="https://docs.superduperdb.com/docs/category/use-cases"><strong>Use-Cases</strong></a> | <a href="https://docs.superduperdb.com/docs/docs/get_started/installation"><strong> Installation</strong></a> | <a href="https://github.com/SuperDuperDB/superduper-community-apps"><strong>Community Apps</strong></a> |  <a href="https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA"><strong> Slack </strong></a> | <a href="https://www.youtube.com/channel/UC-clq9x8EGtQc6MHW0GF73g"><strong> Youtube </strong></a>

</div>


<div align="center">
	<a href="https://github.com/superduperdb/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-green" alt="License - Apache 2.0"></a>	
	<a href="https://pypi.org/project/superduperdb"><img src="https://img.shields.io/pypi/v/superduperdb?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
  	<a href="https://pypi.org/project/superduperdb"><img src="https://img.shields.io/pypi/pyversions/superduperdb.svg" alt="Supported Python versions"></a>    
	<a href="https://github.com/SuperDuperDB/superduperdb/actions/workflows/ci_code.yml"><img src="https://github.com/SuperDuperDB/superduperdb/actions/workflows/ci_code.yml/badge.svg?branch=main" /></a>	
	<a href="https://codecov.io/gh/superduperdb/superduperdb/branch/main"><img src="https://codecov.io/gh/superduperdb/superduperdb/branch/main/graph/badge.svg" alt="Coverage"></a>
	<a href="https://twitter.com/superduperdb" target="_blank"><img src="https://img.shields.io/twitter/follow/nestframework.svg?style=social&label=Follow @SuperDuperDB"></a>

[**English**](README.md) |
<b>中文</b> |
[**日本語**](README_ja.md)

</div>


---

### 📣 我们正式发布SuperDuperDB v0.1版本，包括以下内容:
- 支持主要的SQL数据库：PostgreSQL、MySQL、SQLite、DuckDB、Snowflake、BigQuery、ClickHouse、DataFusion、Druid、Impala、MSSQL、Oracle、pandas、Polars、PySpark和Trino（当然还有MongoDB）
- 改进了文档
- 模块化测试套件

***SuperDuperDB是开源的。请支持我们：***
- 在仓库给我们点亮一颗小星星！⭐
- [在ProductHunt上为我们投票！](https://www.producthunt.com/posts/superduperdb) 🗳️
- 通过在领英上转发[这篇帖子](https://www.linkedin.com/company/superduperdb)和在推特上转发[这条推文](https://twitter.com/superduperdb)，帮助我们传播消息，并标记任何可能对SuperDuperDB感兴趣的人 📣

***开始使用：***
- 查看我们已经实现的使用案例[在这里的文档中](https://docs.superduperdb.com/docs/category/use-cases)，以及社区构建的应用，在专门的[superduper-community-apps仓库](https://github.com/SuperDuperDB/superduper-community-apps)

想了解更多关于SuperDuperDB的信息，以及我们为什么认为它非常需要，请[阅读这篇博客文章](https://blog.superduperdb.com/superduperdb-the-open-source-framework-for-bringing-ai-to-your-datastore/)。

---

## SuperDuperDB是什么？🔮

SuperDuperDB是一个通用的AI开发和部署框架，专为**集成任何机器学习模型**（如PyTorch、Sklearn、HuggingFace）和**AI API**（如OpenAI、Antrophic、Cohere）**直接与您现有的数据库相连**而设计，包括流式推理、模型训练和向量搜索。SuperDuperDB不是另一个新的数据库，它能“超级增强”您目前使用的数据库。

SuperDuperDB消除了复杂的MLOps管道和专用向量数据库的需求，使您能够通过简单的Python接口，高效灵活地构建端到端的AI应用！

- 生成式AI & 基于LLM的Chatbot
- 向量搜索
- 标准的机器学习应用场景（分类、分割、推荐等）
- 涉及到超专业模型的高度定制化AI应用场景

### 核心特性：
- **[将AI与现有数据基础设施集成](https://docs.superduperdb.com/docs/docs/walkthrough/apply_models)：** 在单一可扩展的系统中将任何AI模型和API与您的数据库集成，无需额外的预处理步骤、ETL或冗余代码。
- **[流式推理](https://docs.superduperdb.com/docs/docs/walkthrough/daemonizing_models_with_listeners)：** 当新数据到达时，让您的模型自动激活且立即计算输出，保持您的系统始终更新。
- **[可扩展的模型训练](https://docs.superduperdb.com/docs/docs/walkthrough/training_models)：** 通过查询您的训练数据，在大型、多样化的数据集上训练AI模型。通过内置的计算优化确保最佳性能。
- **[模型调用链](https://docs.superduperdb.com/docs/docs/walkthrough/linking_interdependent_models/)：** 通过连接模型和API，轻松设置复杂的工作流程，以相互依赖和顺序的方式协同工作和调用。
- **[简单易扩展的接口](https://docs.superduperdb.com/docs/docs/fundamentals/procedural_vs_declarative_api)：** 添加并利用Python生态系统中的任何函数、程序、脚本或算法，增强您的工作流和应用。在使用SuperDuperDB时，可以仅通过简单的Python命令即可深入到任何实现层级，包括模型的内部实现。
- **[处理复杂数据类型](https://docs.superduperdb.com/docs/docs/walkthrough/encoding_special_data_types/)：** 在您的数据库中直接处理图像、视频、音频等数据，以及任何可以在Python中编码为`bytes`的类型的数据。
- **[特征存储](https://docs.superduperdb.com/docs/docs/walkthrough/encoding_special_data_types)：** 将您的数据库转变为可用于存储和管理AI模型中任意数据类型的数据输入和输出的中心化存储库，使各种数据可以在熟悉的环境中易结构化的格式来使用。
- **[向量搜索](https://docs.superduperdb.com/docs/docs/walkthrough/vector_search)：** 无需将数据复制和迁移到其他专门的向量数据库 - 将您现有的测试和生产的数据库转变为全功能的多模态向量搜索数据库，包括使用强大的模型和API轻松生成数据的向量Embedding和数据的向量索引。

### 选择SuperDuperDB的理由
|         | SuperDuperDB的优势                                               | 未采用SuperDuperDB的常见挑战                    |
|---------|---------------------------------------------------------------|-----------------------------------------|
| 数据管理与安全 | 数据始终存储在数据库中，AI产生的输出与输入也存储在数据库中，供下游应用使用。数据访问和安全性通过数据库访问管理外部控制。 | 数据复制和迁移到不同环境，以及专门的向量数据库，增加了数据管理负担和安全风险。 |
| 基础设施    | 只需在一个环境内构建、发布和管理AI应用，提高了拓展性和最佳的计算效率。                          | 复杂的碎片化基础设施，包括多个管道，带来高昂的采用和维护成本，增加安全风险。  |
| 代码      | 由于简单和声明式的API，学习曲线极小，仅需简单的Python命令。                            | 需要使用数百行代码, 不同的环境，不同的工具。                 |


## 目前支持的数据库 (*更多的在来的路上*):

<table>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xmongodb.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xmongodb-atlas.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xaws-s3.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xpostgresql.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xmysql.png" width="139px" />
        </td>
    </tr>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xsqlite.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xduckdb.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xsnowflake.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xbigquery.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xclickhouse.png" width="139px" />
        </td>
    </tr>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xdatafusion.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xdruid.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Ximpala.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xmssql.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xoracle.png" width="139px" />
        </td>
    </tr>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xpandas.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xpolars.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xpyspark.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xtrino.png" width="139px" />
        </td>
    </tr>

</table>

**一行命令，将您现有的数据库转化为基于Python的AI开发和部署平台：**

```
db = superduper('mongodb|postgres|sqlite|duckdb|snowflake://<your-db-uri>')
```

## 目前支持的AI框架和模型 (*更多的在来的路上*):

<table>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/frameworks/2Xpytorch.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/frameworks/2Xscikit-learn.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/frameworks/2Xhuggingface-transformers.png" width="139px"/>
        </td>
    </tr>
</table>

**直接将任何AI模型（无论是开源、商业模型还是自行开发的）与您的数据库集成、训练和管理，仅需一个Python命令即可自动在数据库上模型进行计算输出：**

- 安装和部署模型

```
m = db.add(
    <sklearn_model>|<torch_module>|<transformers_pipeline>|<arbitrary_callable>,
    preprocess=<your_preprocess_callable>,
    postprocess=<your_postprocess_callable>,
    encoder=<your_datatype>
)
```

- 模型推理

```
m.predict(X='<input_column>', db=db, select=<mongodb_query>, listen=False|True, create_vector_index=False|True)
```

- 模型训练

```
m.fit(X='<input_column_or_key>', y='<target_column_or_key>', db=db, select=<mongodb_query>|<ibis_query>)
```





## 预置集成的 AI APIs (*更多的在来的路上*):

<table >
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/apis/2Xopenai.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/apis/2Xcohere.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/apis/2Xanthropic.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/apis/jinaai.png" width="139px"/>
        </td>
    </tr>
</table>

**通过一个简单的Python命令，将通过API访问的三方模型和您自己的其他模型集成在一起**


```
m = db.add(
    OpenAI<Task>|Cohere<Task>|Anthropic<Task>|JinaAI<Task>(*args, **kwargs),   # <Task> - Embedding,ChatCompletion,...
)
```




## 架构图

<p align="center">
  <img width="100%" src="docs/hr/static/img/superduperdb.gif">
</p>



## 精选实例

| 名称                                                      | 链接                                                                                                                                                                                                                                               |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 使用ChatGPT与Snowflake聊天                                  | <a href="https://colab.research.google.com/drive/1YXzAuuQdqkWEQKchglxUuAMzNTKLu5rC#scrollTo=0Zf4Unc_fNBp" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>       |
| 使用Mnist和MongoDB进行流式推理                             | <a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/examples/mnist_torch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                   |
| 使用您的SQL数据库进行多模态向量搜索                          | <a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/examples/sql-example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                   |
| 使用CLIP模型连接文本和图像                                  | <a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/examples/multimodal_image_search_clip.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  |
| 使用ChatGTP对您的文档提问                                   | <a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/examples/question_the_docs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>             |
| 使用Vllm对您的文档提问                                      | <a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/examples/question_the_docs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>             |
| 使用Dask和MiniLM模型进行高吞吐量嵌入                        | <a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/examples/sandbox-example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>               |
| Transformers与Scikit之间的迁移学习                         | <a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/examples/transfer_learning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>             |
| 声明式模型链                                                | <a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/examples/transfer_learning_declarative.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 使用CLIP模型搜索视频                                        | <a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/examples/video_search.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                  |
| 使用LibriSpeech和聊天完成的语音助手                          | <a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/examples/voice_memos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                   |



## 安装

#### 1. 通过`pip`安装SuperDuperDB *(~1分钟)* 
```
pip install superduperdb
```

#### 2. 通过Docker安装SuperDuperDB *(~2分钟)*
   - 如果你需要安装Docker, 查看这里的 <a href="https://docs.docker.com/engine/install/">文档</a>.

```
docker run -p 8888:8888 superduperdb/demo:latest
```

## 代码样例

以下是一些简短的介绍，帮助您了解`superduperdb`的工作原理以及它的简易使用方法。您可以访问<a href="https://docs.superduperdb.com">文档</a>了解更多信息。

#### - 部署 ML/AI 模型到你的数据库

在单一环境下自动在你的数据库计算模型的输出结果


```python
import pymongo
from sklearn.svm import SVC

from superduperdb import superduper

# Make your db superduper!
db = superduper(pymongo.MongoClient().my_db)

# Models client can be converted to SuperDuperDB objects with a simple wrapper.
model = superduper(SVC())

# Add the model into the database
db.add(model)

# Predict on the selected data.
model.predict(X='input_col', db=db, select=Collection(name='test_documents').find({'_fold': 'valid'}))
```


#### - 直接从您的数据库训练模型。
仅通过查询您的数据库，无需额外的数据提取和预处理：


```python
import pymongo
from sklearn.svm import SVC

from superduperdb import superduper

# Make your db superduper!
db = superduper(pymongo.MongoClient().my_db)

# Models client can be converted to SuperDuperDB objects with a simple wrapper.
model = superduper(SVC())

# Predict on the selected data.
model.train(X='input_col', y='target_col', db=db, select=Collection(name='test_documents').find({'_fold': 'valid'}))
```

#### - 基于你的数据进行向量搜索
使用你现在喜欢的数据库作为一个向量搜索数据库，涵盖了模型管理和服务

```python
# First a "Listener" makes sure vectors stay up-to-date
indexing_listener = Listener(model=OpenAIEmbedding(), key='text', select=collection.find())

# This "Listener" is linked with a "VectorIndex"
db.add(VectorIndex('my-index', indexing_listener=indexing_listener))

# The "VectorIndex" may be used to search data. Items to be searched against are passed
# to the registered model and vectorized. No additional app layer is required.
db.execute(collection.like({'text': 'clothing item'}, 'my-index').find({'brand': 'Nike'}))
```

#### - 将AI接口集成，与其他模型协同工作。
使用OpenAI、Jina AI、PyTorch或Hugging Face模型作为向量搜索的嵌入模型。


```python
# Create a ``VectorIndex`` instance with indexing listener as OpenAIEmbedding and add it to the database.
db.add(
    VectorIndex(
        identifier='my-index',
        indexing_listener=Listener(
            model=OpenAIEmbedding(identifier='text-embedding-ada-002'),
            key='abstract',
            select=Collection(name='wikipedia').find(),
        ),
    )
)
# The above also executes the embedding model (openai) with the select query on the key.

# Now we can use the vector-index to search via meaning through the wikipedia abstracts
cur = db.execute(
    Collection(name='wikipedia')
        .like({'abstract': 'philosophers'}, n=10, vector_index='my-index')
)
```


#### - 将 Llama2 模型加到SuperDuperDB中


```python
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = Pipeline(
    identifier='my-sentiment-analysis',
    task='text-generation',
    preprocess=tokenizer,
    object=pipeline,
    torch_dtype=torch.float16,
    device_map="auto",
)

# You can easily predict on your collection documents.
model.predict(
    X=Collection(name='test_documents').find(),
    db=db,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200
)
```

#### - 将模型的输出结果作为下游模型的输入

```python
model.predict(
    X='input_col',
    db=db,
    select=coll.find().featurize({'X': '<upstream-model-id>'}),  # already registered upstream model-id
    listen=True,
)
```



## 社区与帮助

#### 如果您遇到任何问题、疑问、意见或想法：
- 加入我们的<a href="https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA">Slack</a>（期待在那里见到您）。
- 浏览<a href="https://github.com/SuperDuperDB/superduperdb/discussions">我们的GitHub讨论区</a>，或者<a href="https://github.com/SuperDuperDB/superduperdb/discussions/new/choose">提出新问题</a>。
- 对<a href="https://github.com/SuperDuperDB/superduperdb/issues/">现有问题</a>进行评论，或者创建<a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">一个新问题</a>。
- 通过<a href="https://docs.google.com/forms/d/e/1FAIpQLScKNyLCjSEWAwc-THjC7NJVDNRxQmVR5ey30VVayPhWOIhy1Q/viewform">这里</a>提供您宝贵的反馈，帮助我们改进SuperDuperDB！
- 通过电子邮件联系我们：`gethelp@superduperdb.com`。
- 随时直接联系维护者或社区志愿者！

## 贡献

#### 贡献的方式多种多样，并不仅限于编写代码。我们欢迎所有形式的贡献，如：

- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">Bug反馈</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">文档优化</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">优化建议</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">期待功能</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">扩展教程和使用案例示例</a>

请查看我们的[贡献指南](CONTRIBUTING.md)了解详细信息。

## 贡献者
#### 感谢这些出色的人们：

<a href="https://github.com/SuperDuperDB/superduperdb/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SuperDuperDB/superduperdb" />
</a>

## 许可证

SuperDuperDB是开源的，并旨在成为一个社区努力的结果，没有您的支持和热情是无法实现的。
它根据Apache 2.0许可证的条款进行分发。对该项目的任何贡献都将受到同样的条款约束。

## 加入我们

我们正在寻找对我们试图解决的问题有兴趣的好人，全职加入我们。查看我们正在招募的职位<a href="https://join.com/companies/superduperdb">这里</a>！
