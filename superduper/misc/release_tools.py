import dataclasses
import os


def get_tags(package):
    """Get all tags from the git repository.

    :param package: The package to get tags for.
    """
    if package == 'superduper':
        os.system('git fetch upstream --tags')
    else:
        os.system('git fetch origin --tags')
    tags = [x for x in os.popen('git tag').read().split('\n') if x.strip()]
    out = []
    for t in tags:
        try:
            out.append(Tag.parse(t))
        except AssertionError as e:
            if 'Unknown VERSION format' in str(e):
                pass
            else:
                raise e
    print('Got tags:', out)
    return out


@dataclasses.dataclass
class Tag:
    """A class to represent a version tag.

    :param major: The major version number.
    :param minor: The minor version number.
    :param patch: The patch version number.
    :param dev: A boolean indicating if this is a development version.
    """

    major: int
    minor: int
    patch: int
    dev: bool = False

    def __repr__(self):
        if self.dev:
            return f'{self.major}.{self.minor}.{self.patch}.dev'
        else:
            return f'{self.major}.{self.minor}.{self.patch}'

    @classmethod
    def parse(cls, tag: str):
        """Parse a version tag.

        :param tag: The tag to parse.
        """
        import re

        isrelease = re.match('^[0-9]+.[0-9]+.[0-9]+$', tag)
        isdev = re.match('^[0-9]+.[0-9]+.[0-9]+.dev$', tag)

        msg = f'Unknown VERSION format: allowed x.y.z and x.y.z.dev; got {tag}'
        assert isrelease is not None or isdev is not None, msg

        split = tag.split('.')

        parts = [int(x) for x in split[:3]]
        dev = False
        if len(split) == 4:
            if split[-1] == 'dev':
                dev = True
            else:
                raise ValueError(
                    f'Unknown VERSION format: allowed x.y.z and x.y.z.dev; got {tag}'
                )
        else:
            dev = False
        return cls(major=parts[0], minor=parts[1], patch=parts[2], dev=dev)

    def __lt__(self, other):
        return other > self

    def __gt__(self, other):
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and (not self.dev and other.dev)
            or self.major == other.major
            and self.minor == other.minor
            and self.patch > other.patch
            or self.major == other.major
            and self.minor > other.minor
            or self.major > other.major
        )

    def __eq__(self, value):
        return (
            self.major == value.major
            and self.minor == value.minor
            and self.patch == value.patch
        )


def check_committed():
    """Check if there are any uncommitted changes in the repository."""
    lines = [x for x in os.popen('git status --porcelain').readlines() if x]
    lines = ''.join(lines)
    assert not lines, f'Uncommitted changes: \n{lines}'


def get_current_version():
    """Get the current version from the VERSION file."""
    with open('VERSION') as f:
        return Tag.parse(f.readlines()[0].strip())


def check_release(package='superduper'):
    """Check if a release is possible.

    :param package: The package to check for a release.
    """
    check_committed()

    import importlib

    package = importlib.import_module(package)
    version = get_current_version()
    imported_version = Tag.parse(package.__version__)

    msg = f'Imported version {version} doesn\'t match VERSION file {imported_version}'
    assert imported_version == version, msg

    msg = f'Version {version} is a development version, cannot release.'
    assert not version.dev, msg

    tags = get_tags(package=package)

    msg = f'Version {version} already exists: here are the versions: {tags}'
    assert version not in tags, msg

    msg = f'Version {version} is not greater than previous tags: {tags}'
    assert all(version > t for t in tags), msg

    print('Confirming release of version:', version)


if __name__ == '__main__':
    import sys

    if not sys.argv[1:]:
        check_release('superduper')
    else:
        check_release(sys.argv[1])
