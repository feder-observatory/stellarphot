# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure.
import os

from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS


def pytest_configure(config):

    config.option.astropy_header = True
    PYTEST_HEADER_MODULES.pop('h5py', None)

    from .version import version
    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = version


# from astropy.tests.helper import enable_deprecations_as_exceptions

## Uncomment the following line to treat all DeprecationWarnings as
## exceptions. For Astropy v2.0 or later, there are 2 additional keywords,
## as follow (although default should work for most cases).
## To ignore some packages that produce deprecation warnings on import
## (in addition to 'compiler', 'scipy', 'pygments', 'ipykernel', and
## 'setuptools'), add:
##     modules_to_ignore_on_import=['module_1', 'module_2']
## To ignore some specific deprecation warning messages for Python version
## MAJOR.MINOR or later, add:
##     warnings_to_ignore_by_pyver={(MAJOR, MINOR): ['Message to ignore']}
# enable_deprecations_as_exceptions()
