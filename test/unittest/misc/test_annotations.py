import pytest

from superduper.base.exceptions import RequiredPackageVersionsNotFound
from superduper.misc.annotations import requires_packages


def test_basic_requires():
    with pytest.raises(RequiredPackageVersionsNotFound) as excinfo:
        requires_packages(['some-package'])
    assert '\n'.join(['some-package']) in str(excinfo.value)


def test_versioned_requires():
    with pytest.raises(RequiredPackageVersionsNotFound) as excinfo:
        requires_packages(
            ['numpy', '10.1.0', '10.1.0'],
            ['requests', None, '0.1.0'],
            ['numpy', '0.0.1', '0.1.0'],
            ['requests', '0.1.0'],
            ['bad-package'],
        )

    print(excinfo.value)
    reqs = [
        'numpy==10.1.0',
        'requests<=0.1.0',
        'numpy>=0.0.1,<=0.1.0',
        'requests<=0.1.0',
        'bad-package',
    ]
    v = str(excinfo.value)
    for r in reqs:
        assert r in v
