---
sidebar_label: Answer question with LLM
filename: answer_question_with_llm.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Answer question with LLM


<Tabs>
    <TabItem value="No-context" label="No-context" default>
        ```python
        
        llm.predict(query)        
        ```
    </TabItem>
    <TabItem value="Prompt" label="Prompt" default>
        ```python
        from superduperdb import model
        from superduperdb.components.graph import Graph, input_node
        
        @model
        def build_prompt(query):
            return f"Translate the sentence into German: {query}"
        
        in_ = input_node('query')
        prompt = build_prompt(query=in_)
        answer = llm(prompt)
        prompt_llm = answer.to_graph("prompt_llm")
        prompt_llm.predict(query)[0]        
        ```
    </TabItem>
    <TabItem value="Context" label="Context" default>
        ```python
        from superduperdb import model
        from superduperdb.components.graph import Graph, input_node
        
        prompt_template = (
            "Use the following context snippets, these snippets are not ordered!, Answer the question based on this context.\n"
            "{context}\n\n"
            "Here's the question: {query}"
        )
        
        
        @model
        def build_prompt(query, docs):
            chunks = [doc["text"] for doc in docs]
            context = "\n\n".join(chunks)
            prompt = prompt_template.format(context=context, query=query)
            return prompt
            
        
        in_ = input_node('query')
        vector_search_results = vector_search_model(query=in_)
        prompt = build_prompt(query=in_, docs=vector_search_results)
        answer = llm(prompt)
        context_llm = answer.to_graph("context_llm")
        context_llm.predict(query)        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="answer_question_with_llm.md" />
