Contributing to ``stellarphot``
-------------------------------

Installation for one-time testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are testing a pre-release version of stellarphot we recommend setting up
a virtual environment and installing stellarphot in this environment.

Only use one of the methods below for making a virtual environment.

Creating an environment with `conda` or `mamba` (use whichever one you have installed)::

    mamba create -n stellarphot-test python=3.11
    mamba activate stellarphot-test
    pip install git+https://github.com/mwcraig/stellarphot.git@update-docs

Creating an environment with `virtualenv`::


    python3 -m venv stellarphot-test
    source stellarphot-test/bin/activate
    pip install git+https://github.com/mwcraig/stellarphot.git@update-docs


Setup
~~~~~

These steps typically only need to be done once.

1. To setup ``stellarphot`` for development, first create your own copy in GitHub
   by forking the repository.
2. Clone your fork to your local machine.
3. Add the main ``stellarphot`` repository as a remote called ``upstream``.

Make a development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These steps typically only need to be done once. If something goes wrong with
the dev environment, you may need to delete it and start over.

1.  Create a new conda environment for ``stellarphot``. You can call it
    whatever you want. For example, you might call it ``stellarphot-dev``.
    The command to create the environment would be:

    .. code-block:: bash

        # The couple of packages listed below take care of some dependencies
        # that would otherwise need to be compiled.
        mamba create -n stellarphot-dev python=3.11 ccdproc batman-package

2.  Activate the environment:

    .. code-block:: bash

        mamba activate stellarphot-dev

3.  Change directory into the folder with your clone of ``stellarphot``.
    Install ``stellarphot`` in development mode with the dependencies needed
    for testing:

    .. code-block:: bash

        pip install -e .[test]

4.  Install the pre-commit hooks -- these are checks run before every commit:

    .. code-block:: bash

        pre-commit install

Steps when Contributing
~~~~~~~~~~~~~~~~~~~~~~~

These typically need to be done every time you make a contribution.

1.  Create a new branch for your contribution that is based on the branch
    ``upstream/main``. Give the branch a descriptive name. For example, if you
    are adding a new feature, you might call the branch ``add-new-feature``. These
    git command to do this would be ``git checkout -b add-new-feature upstream/main``.
2.  Write a test for your contribution. This should be a new test in the
    appropriate ``tests`` directory. If you are adding a new feature, you should add a new
    test file. If you are fixing a bug, you should add a new test to an
    existing test file. The test should be written using the ``pytest`` framework.
    See the ``tests`` directories for examples. **Make sure your test fails before
    you make your contribution.**
3.  Make your contribution. This could be a new feature, a bug fix, or a
    documentation update. **Make sure your test passes after you make your
    contribution.**
4.  Commit your contribution to your branch. Your commit message should be
    short but descriptive. For example, if you are adding a new feature, you might
    use a commit message like ``Add new feature to do X``.
5.  Push your branch to your fork on GitHub.
6.  Create a pull request from your branch to the ``main`` branch of the main
    ``stellarphot`` repository. Give the pull request a descriptive name and
    description. If your pull request fixes an issue, reference the issue in
    the description using the ``#`` symbol. For example, if your pull request
    fixes issue 123, you would write ``Fixes #123`` in the description.
7.  Wait for your pull request to be reviewed. If there are any issues, you
    may need to make additional commits to your branch to address them. If
    you need to make additional commits, make sure you push them to your
    fork on GitHub. The pull request will be updated automatically.

Some specific examples
~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    settings
    moving_code
