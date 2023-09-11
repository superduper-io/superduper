'AI helper functions for text processing.'
import dataclasses as dc
import re

import pandas as pd

search_title = re.compile(r'^\s*(#+)\s*(.*)', re.MULTILINE).search


@dc.dataclass(frozen=True)
class TextChunker:
    stride: int = 5
    window_size: int = 10
    combine: str = ' '
    text_col: str = 'text'

    def __call__(self, src, files, cache):
        return pd.concat([self._slide_window(src, f, cache) for f in files])

    def _slide_window(self, src, file, cache):
        with open(file, 'r') as f:
            df = pd.DataFrame({'text': f.readlines()})

        curr_title = ''
        context, titles = [], []
        n = len(df)

        for i in range(0, n - min(2, self.window_size), self.stride):
            col = df[self.text_col]
            window_text = self.combine.join(col.iloc[i : min(i + self.window_size, n)])
            if m := search_title(window_text):
                curr_title = m.group()
            context.append(window_text)
            titles.append(cache.get((src, curr_title), 'nan'))

        return pd.DataFrame({self.text_col: context, 'src_url': titles})


chunk_file_contents = TextChunker()
