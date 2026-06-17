Trying out test releases
########################

Thanks for being willing to try out a test release! We strongly recommend that you do this in a virtual
environment. If you don't have one set up, you can create one with `conda` or `mamba` (use whichever one you have installed)::

    mamba create -n stellarphot-test python=3.11
    mamba activate stellarphot-test
    pip install --pre stellarphot[all]

``stellarphot[all]`` installs the full interactive experience (the Jupyter
notebook/widget GUI plus exoplanet light-curve fitting). If you only need the
headless/scriptable science engine, install ``stellarphot`` without the
``[all]`` extra; for just the GUI use ``stellarphot[gui]``.
