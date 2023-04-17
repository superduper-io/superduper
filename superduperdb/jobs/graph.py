import networkx


class TaskGraph:
    def __init__(self):
        self._G = networkx.DiGraph()

    def add_node(self, dependencies):
        ...

    def call_node(self, node):
        ...

    def __call__(self, *args, asynchronous=False, **kwargs):
        if not asynchronous:
            nodes = [n for n in self.G.nodes if not n.ancestors]
            while nodes:
                new_nodes = []
                for n in nodes:
                    self.call_node(n)
                    new_nodes.extend([nn for nn in n.children])
                nodes = new_nodes
        else:
            pass