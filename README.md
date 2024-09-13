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
- If you are working with exoplanet transit observations, `stellarphot` can turns the photometry into exoplanet transit light curves (see installation notes below).

## Installation

`stellarphot` requires Python 3.10 or later.

You can install `stellarphot` with either `pip` or `conda`.  If you are interested in `stellarphot` for exoplanet transit light curves, `conda` is recommended at the moment because of an issue with installing one of the dependencies.

- Install with `conda` using
  ```
  conda install -c conda-forge stellarphot
  ```
  If you are interested in exoplanet light curve fitting, also install `batman` using

  ```
  conda install -c conda-forge batman-package
  ```

- Install with `pip` using
  ```
  pip install stellarphot
  ```
  or if you are interested in exoplanet light curve fitting you should instead use:

  ```
  pip install stellarphot[exo_fitting]
  ```

## Getting started with stellarphot

1. Start Jupyterlab from the command line: `jupyter lab`
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

This project is Copyright (c) 2019-2024 The Stellarphot Team and licensed under the terms of the BSD 3-Clause license. This package is based upon the [Astropy package template](https://github.com/astropy/package-template) which is licensed under the BSD 3-clause license.
