version = "unknown.dev"
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    version = _version("stellarphot")
except PackageNotFoundError:
    pass
