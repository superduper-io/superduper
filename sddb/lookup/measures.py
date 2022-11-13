

def dot(x, y):
    return x.matmul(y.T)


def css(x, y):
    x = x.div(x.pow(2).sum(1).sqrt()[:, None])
    y = y.div(y.pow(2).sum(1).sqrt()[:, None])
    return dot(x, y)


measures = {'dot': dot, 'css': css}
