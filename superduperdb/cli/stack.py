from . import command


@command(help='Apply the stack tarball to the database')
def apply(yaml_path: str, identifier: str):
    raise NotImplementedError
