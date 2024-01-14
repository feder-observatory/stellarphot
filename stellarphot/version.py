version = "unknown.dev"
try:
    from importlib_metadata import PackageNotFoundError
    from importlib_metadata import version as _version

    version = _version("stellarphot")
except ImportError:
    from pkg_resources import DistributionNotFound, get_distribution

    try:
        version = get_distribution("stellarphot").version
    except DistributionNotFound:
        pass
except PackageNotFoundError:
    pass
