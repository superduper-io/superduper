import numpy
import pandas
import tqdm


def df_from_iterator(it):
    out = []
    for r in tqdm.tqdm(it):
        out.append(r)
    return pandas.DataFrame(out)


def arrays_from_iterator(it, keys):
    df = df_from_iterator(it)
    out = {}
    for k in keys:
        out[k] = numpy.stack(df[k])
    return out
