from . import command


@command(help='Apply the stack tarball to the database')
def apply(yaml_path: str, identifier: str):
    """Apply the stack tarball to the database.

    :param yaml_path: Path to the stack tarball.
    :param identifier: Stack identifier.
    """
    raise NotImplementedError
