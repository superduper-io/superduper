# Common issues in AI-data development

Traditionally, AI development and databases have lived in separate silo-ed worlds, which 
only interact as an afterthought at the point where a production system is required to 
apply an AI model to a row or table in a database and store and serve the resulting predictions.

Let's see how this can play out in practice.

Suppose our situation is as follows:

- We have data in production populated by users accessing a popular website, and which sends JSON records to MongoDB, with references to web URLs hosted on a separate image server.
- Each record contains some data left behind by users which may be useful for training a classification model.

Given this data, we would like to accomplish the following:

- We would like to use our data hosted in MongoDB to train a model to classify images
- We want to use the probabilistic estimates for the classifications in a production scenario

To do this, we need to be able to implement these high level steps:

- Access the images and data in a way enabling training a computer-vision classification model
- Train the model on the accessible images and associated labels
- Once the model is trained, deploy it in a way so that incoming user data's images are classified using his model in as timely a manner as possible.
- Consume the outputs of the model in the functionality of the website

Pre-2023, this is an extremely arduous task. In order to get a model working using this data, and working with this deployment, we would typically be required to perform something equivalent to the following sequence of tasks (module cloud provider, exact software choices). 

1. Download a snapshot of his data from MongoDB and place in an AWS s3 bucket.
2. Write a script to run through all of the images mentioned in the BSON records downloaded, and download these to a fast-access hard drive, in elastic block storage
3. Write a new script which processes the data downloaded from MongoDB, extracting a dataframe of labels and image URIs in s3. He has to take care not to make book-keeping errors in the process.
4. Prepare the model for training, using for example, `torchvision`, to preprocess the images for batching using GPUs, and `torch` for writing the model forward pass.
5. To perform training, spin up an EC2 instance, or use AWS Sagemaker. Often the lock-in nature of AWS Sagemaker staves off a large percentage of users. This means defining AWS Cloudformation templates allowing us to easily start a training instance, mount the hard-drive containing the images, and stop the instance with an AWS lambda function after completion.
6. If the model is declared sufficient, we move to building a production pipeline. Again, to avoid, vendor lock-in, we might opt for the open-source Apache Airflow. We build a DAG using Airflow, which periodically checks for records which are have yet to be classified in MongoDB, loads data from the database and dumps this into an s3 bucket, downloads the images referred to in this data, loads the model and applies preprocessing to the images, followed by running the model over this data, and finally applying post-processing to the outputs. The classifications are then made human readable, looking up indices in a lookup table we provide. The classifications are finally inserted back to MongoDB, along with the probabilistic estimates from the PyTorch model. 
7. In performing 6., we are required to provide our model in a way which may be consumed by the production system. We defines a new inference only preprocessor which may be used by the `torch` model, writes a script which re-instantiates his model from the parameters applied in training, and also an additional script responsible for the bookkeeping between the `forward` pass outputs, and the human readable probabilistic predictions.

This story may sound super familiar to AI developers and data scientists. It can cause delays of months or longer, in deploying even standard use-cases to production. There are indeed tools out there which smoothen the journey to consuming AI through the database - Zen ML, Comet ML, etc.. However, these simply make each step in this complex sequence easier to execute. 

What if, by shifting the focus from model centricity, to database centricity, we could simplify matters considerably?

This is where SuperDuperDB comes in. Let's look at how SuperDuperDB might allow Jim to work:

1. We register the "user" collection in MongoDB with SuperDuperDB, configuring the fact that the URLs point to image 
   URIs with SuperDuperDB's inbuilt encoder system. This induces SuperDuperDB to spring into action every time data is updated
   in the user collection. SuperDuperDB automatically downloads the URIs to MongoDB and visible to the models installed in 
   SuperDuperDB, ready for training and inference.
2. We program `preprocess` and `postprocess` python functions on his class and wrap these together with the PyTorch model with a single wrapper `superduper`. 
   We import a SuperDuperDB client, and pass the client and a MongoDB style query `q = collection.find({'img': {'$exists': 1}})` 
   to the `.fit` method of the wrapped model.
3. SuperDuperDB springs into action, uploading the model to SuperDuperDB, and triggering model training on SuperDuperDB's `dask` worker pool.
   Once finished, metrics and model-state are preserved in the configured artifact store.
4. Using, one command, `model.predict('img', select=q, watch=True)`, Jim installs the model on the user collection, 
   so that as new data are inserted, the model is evaluated in inference model, 
   the predictions postprocessed, and human readable outputs are inserted to the user collection.
   
With SuperDuperDB setup in this way, and models configured to operate on the "user" collection, 
the deployment reacts automatically to changes in the "user" collection and 
model outputs are continuously integrated back into the database.

