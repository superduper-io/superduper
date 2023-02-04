
def dot(x, y):
    return x.matmul(y.T)


def css(x, y):
    x = x.div(x.norm(dim=1)[:, None])
    y = y.div(y.norm(dim=1)[:, None])
    return dot(x, y)
