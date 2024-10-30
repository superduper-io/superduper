# TODO do we need this?
def code(my_callable):
    """Decorator to mark a function as remote code.

    :param my_callable: The callable to mark as remote code.
    """
    my_callable.is_remote_code = True
    return my_callable
