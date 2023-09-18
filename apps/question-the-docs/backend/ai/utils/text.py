'AI helper functions for text processing.'
import dataclasses as dc
import re

import pandas as pd

CODE_BLOCK_PREFIX = ('python', 'ruby', 'perl', 'bash')


@dc.dataclass(frozen=True)
class TextChunker:
    stride: int = 5
    window_size: int = 10
    combine: str = ' '
    text_col: str = 'text'

    def __call__(self, src, files, cache):
        return pd.concat([self._slide_window(src, f, cache) for f in files])

    def _get_code_block_lines(self, lines, non_ended_code_block):
        code_start = -1
        code_end = -1
        for i, line in enumerate(lines):
            if any([lang for lang in CODE_BLOCK_PREFIX if f'```{lang}' in line]):
                code_start = i
            elif line.startswith('```'):
                code_end = i

        if code_end != -1:
            non_ended_code_block = False

        if code_end == -1 and code_start != -1:
            non_ended_code_block = True

        window_text = self.combine.join(lines)

        if code_start == -1 and code_end == -1:
            if non_ended_code_block:
                excluded_text = ''
            else:
                excluded_text = window_text
        else:
            code_end = len(lines) if code_end == -1 else code_end
            code_start = 0 if code_start == -1 else code_start

            lines = lines[0:code_start] + lines[code_end + 1 :]
            excluded_text = self.combine.join(lines)

        return window_text, excluded_text, non_ended_code_block

    def _slide_window(self, src, file, cache):
        with open(file, 'r') as f:
            df = pd.DataFrame({'text': f.readlines()})

        curr_title = ''
        context, titles = [], []
        n = len(df)
        non_ended_previous_code_block = False

        for i in range(0, n - min(2, self.window_size), self.stride):
            col = df[self.text_col]
            lines = col.iloc[i : min(i + self.window_size, n)].values.tolist()

            (
                window_text,
                excluded_text,
                non_ended_previous_code_block,
            ) = self._get_code_block_lines(lines, non_ended_previous_code_block)

            if excluded_text:
                title = re.findall(r'^\s*(#+)\s*(.*)', excluded_text, re.MULTILINE)
                if title:
                    curr_title = title[0][-1]

            context.append(window_text)
            titles.append(cache.get((src, curr_title), 'nan'))

        return pd.DataFrame({self.text_col: context, 'src_url': titles})


chunk_file_contents = TextChunker()
