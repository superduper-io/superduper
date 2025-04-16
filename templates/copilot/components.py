from concurrent.futures import ThreadPoolExecutor
import datetime
from superduper import trigger, CFG


from litellm import completion
from superduper import Model
import typing as t

from superduper.components.cdc import CDC
from superduper import logging


CHANGE_PROMPT = """
You are a code reviewer. Please review the following code and provide feedback in 1-3 sentences.
If the code contains a diff, please focus on the changes reflected in the diff.
If the code does not contain a diff, please focus on the entire code.
If the code is "good enough", please write an empty string (no characters) - DO NOT WRITE "It seems to be working as intended...".
HERE IS THE CONTENT:
--------------------
"""

ASK_PROMPT = """
Here is a question about the project. Please answer with the context of the project
as the most important aspect. The files and content included in the prompt with 
the markers: ==========FILE========= and ==========CONTENT==========

HERE IS THE QUESTION:
---------------------
"""


class AskSuperduper(Model):
    model: str = "gpt-4-turbo"
    project: str
    prompt: str = ASK_PROMPT

    def build_prompt_from_file_context(self, files: t.Dict[str, str]):
        prompt = """"""
        for filename, content in files.items():
            prompt += f"==========FILE=========: {filename}\n"
            prompt += f"==========CONTENT==========:\n{content}\n\n"
            prompt += "===========================\n\n"

        prompt += "HERE IS THE QUESTION:\n==================\n"
        return prompt

    def predict(self, question: str, filename: str):
        """Predict the answer to a question based on the content of a file.
        
        :param question: The question to ask.
        :param filename: The name of the file to analyse.
        :param project_name: The name of the project.
        """

        files = self.db[f'files_{self.project}'].execute()
        files = {r['filename']: r['content'] for r in files}

        question = self.build_prompt_from_file_context(files) + question

        messages = [
            {
                "role": "user",
                "content": self.prompt + question,
            }
        ]
        output = completion(self.model, messages=messages)

        # # get the first message from the response
        response = output['choices'][0]['message']['content']
        return response



class Commenter(CDC):
    """A component that comments on code changes using a language model.
    
    :param prompt: The prompt to use for the language model.
    :param model: The model to use for the language model.
    """
    prompt: str = CHANGE_PROMPT
    model: str = "gpt-4-turbo"

    @trigger('apply', 'insert', 'update')
    def analyse_new(self, ids: t.List[str] | None = None):
        logging.info('Analysing new files with ids:', ids)
        if ids is None:
            data = self.db[self.cdc_table].execute()
        else:
            data = self.db[self.cdc_table].subset(ids)

        comments = []

        def process_record(r):
            logging.info('Analysing:- ', r['filename'])
            response = self.analyse(r['diff'] if r['diff'] else r['content']).strip()
            logging.info('Analysing:- ', r['filename'], '... DONE')
            if response:
                return {
                    'filename': r['filename'],
                    'comment': response,
                    'analysed': str(datetime.datetime.now()),
                }

        with ThreadPoolExecutor() as executor:
            results = executor.map(process_record, data)

        for result in results:
            if result:
                comments.append(result)

        self.db[f'comments_{self.identifier}'].insert(comments)

        logging.info('Analysing new files with ids:', ids, '... DONE')


    def analyse(self, content: str):
        """Analyse the content of a file and return a comment.
        
        :param content: The content of the file to analyse.
        """

        messages = [
            {
                "role": "user",
                "content": self.prompt + 
                    '\nHERE IS THE CONTENT:\n-------------\n' + content,
            }
        ]
        output = completion(self.model, messages=messages)

        # get the first message from the response
        response = output['choices'][0]['message']['content']
        return response
