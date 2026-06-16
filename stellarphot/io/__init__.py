# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This package intentionally does not re-export the contents of its submodules.
# Import directly from stellarphot.io.aavso, stellarphot.io.aij, or
# stellarphot.io.tess. Keeping this __init__ empty also avoids a core <-> io
# import cycle: stellarphot.io.tess imports stellarphot.core, so importing
# stellarphot.io.aavso from core.py must not pull in tess.
