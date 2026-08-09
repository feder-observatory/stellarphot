# Import from .core lazily (PEP 562): the GUI import chain reaches
# ``stellarphot.transit_fitting.io``, which runs this file on the way, and
# .core imports pytransit -- seconds of import time plus warnings that end
# up in notebook cells that never do any transit fitting.
__all__ = ["TransitModelFit", "TransitModelOptions"]


def __getattr__(name):
    if name in __all__:
        from . import core

        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
