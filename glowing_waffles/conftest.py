# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure.
import os

try:
    # When the pytest_astropy_header package is installed
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)

    def pytest_configure(config):
        config.option.astropy_header = True
except ImportError:
    # TODO: Remove this when astropy 2.x and 3.x support is dropped.
    # Probably an old pytest-astropy package where the pytest_astropy_header
    # is not a dependency.
    try:
        from astropy.tests.plugins.display import (pytest_report_header,
                                                   PYTEST_HEADER_MODULES,
                                                   TESTED_VERSIONS)
    except ImportError:
        # TODO: Remove this when astropy 2.x support is dropped.
        # If that also did not work we're probably using astropy 2.0
        from astropy.tests.pytest_plugins import (pytest_report_header,
                                                  PYTEST_HEADER_MODULES,
                                                  TESTED_VERSIONS)

try:
    # TODO: Remove this when astropy 2.x support is dropped.
    # This is the way to get plugins in astropy 2.x
    from astropy.tests.pytest_plugins import *
except ImportError:
    # Otherwise they are installed as separate packages that pytest
    # automagically finds.
    pass


from astropy.tests.helper import enable_deprecations_as_exceptions

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

# Customize the following lines to add/remove entries from
# the list of packages for which version numbers are displayed when running
# the tests. Making it pass for KeyError is essential in some cases when
# the package uses other astropy affiliated packages.
try:
    PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
    del PYTEST_HEADER_MODULES['h5py']
    del PYTEST_HEADER_MODULES['astropy-helpers']
except KeyError:
    pass

# This is to figure out the package version, rather than
# using Astropy's
from .version import version

packagename = os.path.basename(os.path.dirname(__file__))
TESTED_VERSIONS[packagename] = version
