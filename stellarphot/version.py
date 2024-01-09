version = "unknown.dev"
try:
    from importlib_metadata import PackageNotFoundError
    from importlib_metadata import version as _version

    version = _version("my-package")
except ImportError:
    from pkg_resources import DistributionNotFound, get_distribution

    try:
        version = get_distribution("my-package").version
    except DistributionNotFound:
        pass
except PackageNotFoundError:
    pass
