from rich.tree import Tree


def dict_to_tree(dictionary, root: str = 'root', tree=None):
    """
    Convert a dictionary to a `rich.Tree`.

    :param dictionary: Input dict
    :param root: Name of root
    :param tree: Ignore
    """
    if tree is None:
        tree = Tree(root)

    for key, value in dictionary.items():
        if isinstance(value, dict):
            # If the value is another dictionary, create a subtree
            subtree = tree.add(f"[bold yellow]{key}")
            dict_to_tree(value, root=root, tree=subtree)
        else:
            # Add the key and value as a leaf node
            if value == 'breaking':
                tree.add(f"[bold cyan]{key}: [red]{value}")
            elif value == 'update':
                tree.add(f"[bold cyan]{key}: [blue]{value}")
            else:
                tree.add(f"[bold cyan]{key}: [green]{value}")

    return tree


if __name__ == '__main__':
    data = {
        "sub": {"status": "broken", "changes": {"b": 4}},
        "test": {"status": "update", "changes": {"sub": "?sub"}},
    }

    from rich.console import Console

    console = Console()
    tree = dict_to_tree(data)
    console.print(tree)
