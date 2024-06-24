---
sidebar_label: Get LLM Finetuning Data
filename: get_llm_finetuning_data.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Get LLM Finetuning Data

The following are examples of training data in different formats.


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from datasets import load_dataset
        from superduperdb.base.document import Document
        dataset_name = "timdettmers/openassistant-guanaco"
        dataset = load_dataset(dataset_name)
        
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        
        train_documents = [
            Document({**example, "_fold": "train"})
            for example in train_dataset
        ]
        eval_documents = [
            Document({**example, "_fold": "valid"})
            for example in eval_dataset
        ]
        
        datas = train_documents + eval_documents        
        ```
    </TabItem>
    <TabItem value="Prompt-Response" label="Prompt-Response" default>
        ```python
        from datasets import load_dataset
        from superduperdb.base.document import Document
        dataset_name = "mosaicml/instruct-v3"
        dataset = load_dataset(dataset_name)
        
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        
        train_documents = [
            Document({**example, "_fold": "train"})
            for example in train_dataset
        ]
        eval_documents = [
            Document({**example, "_fold": "valid"})
            for example in eval_dataset
        ]
        
        datas = train_documents + eval_documents        
        ```
    </TabItem>
    <TabItem value="Chat" label="Chat" default>
        ```python
        from datasets import load_dataset
        from superduperdb.base.document import Document
        dataset_name = "philschmid/dolly-15k-oai-style"
        dataset = load_dataset(dataset_name)['train'].train_test_split(0.9)
        
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        
        train_documents = [
            Document({**example, "_fold": "train"})
            for example in train_dataset
        ]
        eval_documents = [
            Document({**example, "_fold": "valid"})
            for example in eval_dataset
        ]
        
        datas = train_documents + eval_documents        
        ```
    </TabItem>
</Tabs>
We can define different training parameters to handle this type of data.


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        # Function for transformation after extracting data from the database
        transform = None
        key = ('text')
        training_kwargs=dict(dataset_text_field="text")        
        ```
    </TabItem>
    <TabItem value="Prompt-Response" label="Prompt-Response" default>
        ```python
        # Function for transformation after extracting data from the database
        def transform(prompt, response):
            return {'text': prompt + response + "</s>"}
        
        key = ('prompt', 'response')
        training_kwargs=dict(dataset_text_field="text")        
        ```
    </TabItem>
    <TabItem value="Chat" label="Chat" default>
        ```python
        # Function for transformation after extracting data from the database
        transform = None
        
        key = ('messages')
        training_kwargs=None        
        ```
    </TabItem>
</Tabs>
Example input_text and output_text


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        data = datas[0]
        input_text, output_text = data["text"].rsplit("### Assistant: ", maxsplit=1)
        input_text += "### Assistant: "
        output_text = output_text.rsplit("### Human:")[0]
        print("Input: --------------")
        print(input_text)
        print("Response: --------------")
        print(output_text)        
        ```
    </TabItem>
    <TabItem value="Prompt-Response" label="Prompt-Response" default>
        ```python
        data = datas[0]
        input_text = data["prompt"]
        output_text = data["response"]
        print("Input: --------------")
        print(input_text)
        print("Response: --------------")
        print(output_text)        
        ```
    </TabItem>
    <TabItem value="Chat" label="Chat" default>
        ```python
        data = datas[0]
        messages = data["messages"]
        input_text = messages[:-1]
        output_text = messages[-1]["content"]
        print("Input: --------------")
        print(input_text)
        print("Response: --------------")
        print(output_text)        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="get_llm_finetuning_data.md" />
