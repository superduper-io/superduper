def str_shape(shape):
    return str(shape[0]) if len(shape) == 0 else 'x'.join([str(x) for x in shape])
