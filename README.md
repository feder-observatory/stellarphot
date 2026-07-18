# A Python Package for Stellar Photometry

[![Powered by Astropy Badge](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)
[![GitHub Workflow badge](https://github.com/feder-observatory/stellarphot/workflows/Test/badge.svg?branch=main)](https://github.com/feder-observatory/stellarphot/actions?query=workflow%3ATest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/feder-observatory/stellarphot/main.svg)](https://results.pre-commit.ci/latest/github/feder-observatory/stellarphot/main)
[![codecov](https://codecov.io/gh/feder-observatory/stellarphot/graph/badge.svg?token=uVrdNencSQ)](https://codecov.io/gh/feder-observatory/stellarphot)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10679636.svg)](https://doi.org/10.5281/zenodo.10679636)

Stellarphot is a Python package to allow you to turn reduced astronomical images of point sources (e.g. stars) into
useful photometry, with a focus on variable star and exoplanet transit observations.  Specifically:

- If you have reduced astronomical images as FITS files but haven't obtained photometry yet, `stellarphot` can perform aperture photometry on your images.
- If you already have aperture photometry for a field, `stellarphot` can
  - choose comparison stars based on a catalog (e.g. APASS DR9),
  - calculate relative flux (like [AstroImageJ](https://www.astro.louisville.edu/software/astroimagej/)),
  - calculate calibrated magnitudes by transforming to a catalog (e.g. APASS DR9), and/or
  - calculate calibrated magnitudes with a user-provided set of comparison stars (as is done in AAVSO submissions).
  - export ensemble photometry results in the AAVSO Extended File Format for direct upload to WebObs.
- If you are working with exoplanet transit observations, `stellarphot` can turns the photometry into exoplanet transit light curves (see installation notes below).

## Installation

`stellarphot` requires Python 3.11 or later. You can install `stellarphot` using `uv`, `pixi`, `pip` or `conda`. 

Both `pixi` and `uv` work on a model where you create a working directory for your project and install the packages into that working directory.  This is a good way to keep your work organized and avoid conflicts with other Python packages you may have installed. You an examine the documentation for [`uv`](https://uv.readthedocs.io/en/latest/) or [`pixi`](https://pixi.readthedocs.io/en/latest/) for details on how to use those tools.

`conda` and `pip` work on a model where you install the packages into your existing Python environment. This is a good way to get started quickly, but you may run into conflicts with other packages you have installed.


### Installing with `uv`

- You can create a working environment for `stellarphot` using `uv` with the following commands which initialize a new project directory (replace `project_dir_name` with the name of your choice) and install the package with all optional dependencies:
    ```
    uv init project_dir_name
    cd project_dir_name
    uv add "stellarphot[all]"
    ```
    This installs the entire package with all optional dependencies. Once that is done, you can start JupyterLab interface with `uv run jupyter lab`.

    The optional dependencies are grouped into extras so you can install only what you need after initializing the `uv` environment and switching to the working directory:

    - `uv add stellarphot` — base install. The headless/scriptable science
        engine (data structures, photometry, catalog access). Does **not** include the
        Jupyter/widget GUI.
    - `uv add "stellarphot[gui]"` — adds the notebook/widget interface.
    - `uv add "stellarphot[exoplanet]"` — adds exoplanet transit light-curve
        fitting (`pytransit`).


### Installing with `pixi`

- You can create a working environment for `stellarphot` using `pixi` with the following commands which initialize a new project directory (replace `project_dir_name` with the name of your choice) and install the package with all optional dependencies:
    ```
    pixi init project_dir_name
    cd project_dir_name
    pixi add stellarphot
    ```
    This installs the entire package with all optional dependencies. Once that is done, you can start JupyterLab interface with `pixi run jupyter lab`. [**NOTE**: Since `pixi` uses `conda-forge` as its default channel, it doesn't always have the latest version of `stellarphot`. If you want the latest version, use `pixi add --pypi "stellarphot[all]"`].
    
    The optional dependencies are grouped into extras so you can install only what you need after initializing the `pixi` environment and switching to the working directory:
    - `pixi add --pypi "stellarphot[all]"` — complete install pulled from PyPI with all optional dependencies. 
        Jupyter/widget GUI.
    - `pixi add --pypi "stellarphot"` — base install. The headless/scriptable science
        engine (data structures, photometry, catalog access). Does **not** include the
        Jupyter/widget GUI.
    - `pixi add --pypi "stellarphot[gui]"` — adds the notebook/widget interface.
    - `pixi add --pypi "stellarphot[exoplanet]"` — adds exoplanet transit light-curve
        fitting (`pytransit`).


### Installing with `conda`

- Install with `conda` using
  ```
  conda install -c conda-forge stellarphot
  ```
  This installs the entire package with all optional dependencies. Then you can start JupyterLab interface with `jupyter lab`.  There is no easy way to install only some of the optional dependencies with `conda` at this time. [**NOTE**: `conda-forge` doesn't always have the latest version of `stellarphot`. If you want the latest version, use `uv`, `pixi`, or `pip` instead.]

### Installing with `pip`

- Install with `pip`. Most people want the full interactive experience (the
  Jupyter notebooks and widgets plus exoplanet fitting):
  ```
  pip install stellarphot[all]
  ```
  The optional dependencies are grouped into extras so you can install only what you need:

  - `pip install stellarphot` — base install. The headless/scriptable science
    engine (data structures, photometry, catalog access). Does **not** include the Jupyter/widget GUI.
  - `pip install stellarphot[gui]` — adds the notebook/widget interface.
  - `pip install stellarphot[exoplanet]` — adds exoplanet transit light-curve
    fitting (`pytransit`).
  - `pip install stellarphot[all]` — everything above.

## Getting started with stellarphot

1. Start Jupyterlab from the command line: `jupyter lab` (or `uv run jupyter lab` or `pixi run jupyter lab` if you installed with `uv` or `pixi`).
2. Once JupyterLab opens in your web browser, open the Launcher (see Figure below)
3. Click on the notebook you want (see figure below) and follow the instructions in the notebook.
Output files and the settings used to generate them will show up in the file browser

<img width="833" alt="stellarphot-screenshot" src="https://github.com/feder-observatory/stellarphot/blob/bd5e08dca6e390239663bf4d4db797d84abf603c/docs/_static/launcher.png">

## Questions?

Feel free to contact @mwcraig or @JuanCab with your questions about using `stellarphot`.

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://mwcraig.github.io"><img src="https://avatars.githubusercontent.com/u/1147167?v=4?s=100" width="100px;" alt="Matt Craig"/><br /><sub><b>Matt Craig</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/issues?q=author%3Amwcraig" title="Bug reports">🐛</a> <a href="https://github.com/feder-observatory/stellarphot/commits?author=mwcraig" title="Code">💻</a> <a href="#design-mwcraig" title="Design">🎨</a> <a href="#ideas-mwcraig" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-mwcraig" title="Maintenance">🚧</a> <a href="#mentoring-mwcraig" title="Mentoring">🧑‍🏫</a> <a href="#research-mwcraig" title="Research">🔬</a> <a href="https://github.com/feder-observatory/stellarphot/pulls?q=is%3Apr+reviewed-by%3Amwcraig" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/feder-observatory/stellarphot/commits?author=mwcraig" title="Tests">⚠️</a> <a href="#tutorial-mwcraig" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://web.mnstate.edu/cabanela/"><img src="https://avatars.githubusercontent.com/u/1940512?v=4?s=100" width="100px;" alt="Juan Cabanela"/><br /><sub><b>Juan Cabanela</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/issues?q=author%3AJuanCab" title="Bug reports">🐛</a> <a href="https://github.com/feder-observatory/stellarphot/commits?author=JuanCab" title="Code">💻</a> <a href="#design-JuanCab" title="Design">🎨</a> <a href="#ideas-JuanCab" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-JuanCab" title="Maintenance">🚧</a> <a href="https://github.com/feder-observatory/stellarphot/pulls?q=is%3Apr+reviewed-by%3AJuanCab" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/feder-observatory/stellarphot/commits?author=JuanCab" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/madelyn914"><img src="https://avatars.githubusercontent.com/u/56169991?v=4?s=100" width="100px;" alt="madelyn914"/><br /><sub><b>madelyn914</b></sub></a><br /><a href="#ideas-madelyn914" title="Ideas, Planning, & Feedback">🤔</a> <a href="#userTesting-madelyn914" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AbigaleMoen"><img src="https://avatars.githubusercontent.com/u/112969124?v=4?s=100" width="100px;" alt="Abby"/><br /><sub><b>Abby</b></sub></a><br /><a href="#userTesting-AbigaleMoen" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MDeRung2021"><img src="https://avatars.githubusercontent.com/u/90003875?v=4?s=100" width="100px;" alt="MDeRung2021"/><br /><sub><b>MDeRung2021</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/issues?q=author%3AMDeRung2021" title="Bug reports">🐛</a> <a href="#userTesting-MDeRung2021" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Tanner728"><img src="https://avatars.githubusercontent.com/u/90003838?v=4?s=100" width="100px;" alt="Tanner Weyer"/><br /><sub><b>Tanner Weyer</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/commits?author=Tanner728" title="Code">💻</a> <a href="https://github.com/feder-observatory/stellarphot/pulls?q=is%3Apr+reviewed-by%3ATanner728" title="Reviewed Pull Requests">👀</a> <a href="#userTesting-Tanner728" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/WatsonEmily11"><img src="https://avatars.githubusercontent.com/u/99451181?v=4?s=100" width="100px;" alt="Emily Watson"/><br /><sub><b>Emily Watson</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/issues?q=author%3AWatsonEmily11" title="Bug reports">🐛</a> <a href="#userTesting-WatsonEmily11" title="User Testing">📓</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Mamizou"><img src="https://avatars.githubusercontent.com/u/35544119?v=4?s=100" width="100px;" alt="Adam Kline"/><br /><sub><b>Adam Kline</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/commits?author=Mamizou" title="Code">💻</a> <a href="#userTesting-Mamizou" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Sondanaa"><img src="https://avatars.githubusercontent.com/u/9082828?v=4?s=100" width="100px;" alt="Elias Holte"/><br /><sub><b>Elias Holte</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/commits?author=Sondanaa" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/pllim/"><img src="https://avatars.githubusercontent.com/u/2090236?v=4?s=100" width="100px;" alt="P. L. Lim"/><br /><sub><b>P. L. Lim</b></sub></a><br /><a href="#maintenance-pllim" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/clkotnik"><img src="https://avatars.githubusercontent.com/u/26798912?v=4?s=100" width="100px;" alt="clkotnik"/><br /><sub><b>clkotnik</b></sub></a><br /><a href="#ideas-clkotnik" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/meyerpa"><img src="https://avatars.githubusercontent.com/u/14203298?v=4?s=100" width="100px;" alt="Paige Meyer"/><br /><sub><b>Paige Meyer</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/commits?author=meyerpa" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/stottsco"><img src="https://avatars.githubusercontent.com/u/14881940?v=4?s=100" width="100px;" alt="stottsco"/><br /><sub><b>stottsco</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/commits?author=stottsco" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/isobelsnellenberger"><img src="https://avatars.githubusercontent.com/u/36014334?v=4?s=100" width="100px;" alt="Isobel Snellenberger"/><br /><sub><b>Isobel Snellenberger</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/commits?author=isobelsnellenberger" title="Code">💻</a> <a href="#userTesting-isobelsnellenberger" title="User Testing">📓</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://stefannelson.github.io"><img src="https://avatars.githubusercontent.com/u/9082855?v=4?s=100" width="100px;" alt="Stefan Nelson"/><br /><sub><b>Stefan Nelson</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/commits?author=stefannelson" title="Code">💻</a> <a href="#userTesting-stefannelson" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/walkerna22"><img src="https://avatars.githubusercontent.com/u/4551626?v=4?s=100" width="100px;" alt="Nathan Walker"/><br /><sub><b>Nathan Walker</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/commits?author=walkerna22" title="Code">💻</a> <a href="#userTesting-walkerna22" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/janeglanzer"><img src="https://avatars.githubusercontent.com/u/21367441?v=4?s=100" width="100px;" alt="Jane Glanzer"/><br /><sub><b>Jane Glanzer</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/commits?author=janeglanzer" title="Code">💻</a> <a href="#userTesting-janeglanzer" title="User Testing">📓</a></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td align="center" size="13px" colspan="7">
        <img src="https://raw.githubusercontent.com/all-contributors/all-contributors-cli/1b8533af435da9854653492b1327a23a4dbd0a10/assets/logo-small.svg">
          <a href="https://all-contributors.js.org/docs/en/bot/usage">Add your contributions</a>
        </img>
      </td>
    </tr>
  </tfoot>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


## License

This project is Copyright (c) 2019-2026 The Stellarphot Team and licensed under the terms of the BSD 3-Clause license. This package is based upon the [Astropy package template](https://github.com/astropy/package-template) which is licensed under the BSD 3-clause license.
