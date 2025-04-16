from concurrent.futures import ThreadPoolExecutor
import datetime
from superduper import trigger
from superduper.base import Base
from litellm import completion
import typing as t

from superduper.components.cdc import CDC
from superduper import logging


PROMPT = """
You are a code reviewer. Please review the following code and provide feedback in 1-3 sentences.
If the code contains a diff, please focus on the changes reflected in the diff.
If the code does not contain a diff, please focus on the entire code.
If the code is "good enough", please write an empty string (no characters) - DO NOT WRITE "It seems to be working as intended...".
HERE IS THE CONTENT:
-------------
"""


class Commenter(CDC):

    prompt: str = PROMPT
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


