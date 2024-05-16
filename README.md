# A package for transforming stellar photometry

[![Powered by Astropy Badge](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org) [![GitHub Workflow badge](https://github.com/feder-observatory/stellarphot/workflows/Test/badge.svg?branch=main)](https://github.com/feder-observatory/stellarphot/actions?query=workflow%3ATest) [![codecov](https://codecov.io/gh/feder-observatory/stellarphot/graph/badge.svg?token=uVrdNencSQ)](https://codecov.io/gh/feder-observatory/stellarphot) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10679636.svg)](https://doi.org/10.5281/zenodo.10679636)


+ If you already have aperture photometry for a field,
    * calculate relative flux (like [AstroImageJ](https://www.astro.louisville.edu/software/astroimagej/)), and/or
    * calculate calibrated magnitudes by transforming to a catalog (e.g. APASS DR9)
+ If you have calibrated images but haven't done photometry yet, you can do aperture photometry on your images.
+ If you have not calibrated your images, `stellarphot` can help -- choose the


## Installation

You can install `stellarphot` with either pip or conda. `conda` is recommended at the moment because of an issue with installing one of the dependencies of `stellarphot` in Python 3.9 or higher

Install with `conda`:

```
conda install -c conda-forge stellarphot reducer
pip install astronbs
```

or install with `pip`:

```
pip install stellarphot reducer astronbs
```

## Running stellarphot

1. Start jupyterlab from the command line: `jupyter lab`
2. Open the Launcher (see below)
3. Click on the notebook you want (see below)
4. Output files will show up in the file browser
5. Questions? Feel free to contact @mwcraig

<img width="833" alt="stellarphot-screenshot" src="https://user-images.githubusercontent.com/1147167/200139186-100934ca-6d1e-46f9-ac89-a83d05528bb2.png">


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://mwcraig.github.io"><img src="https://avatars.githubusercontent.com/u/1147167?v=4?s=100" width="100px;" alt="Matt Craig"/><br /><sub><b>Matt Craig</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/issues?q=author%3Amwcraig" title="Bug reports">🐛</a> <a href="https://github.com/feder-observatory/stellarphot/commits?author=mwcraig" title="Code">💻</a> <a href="#design-mwcraig" title="Design">🎨</a> <a href="#ideas-mwcraig" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-mwcraig" title="Maintenance">🚧</a> <a href="#mentoring-mwcraig" title="Mentoring">🧑‍🏫</a> <a href="#research-mwcraig" title="Research">🔬</a> <a href="https://github.com/feder-observatory/stellarphot/pulls?q=is%3Apr+reviewed-by%3Amwcraig" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/feder-observatory/stellarphot/commits?author=mwcraig" title="Tests">⚠️</a> <a href="#tutorial-mwcraig" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://web.mnstate.edu/cabanela/"><img src="https://avatars.githubusercontent.com/u/1940512?v=4?s=100" width="100px;" alt="Juan Cabanela"/><br /><sub><b>Juan Cabanela</b></sub></a><br /><a href="https://github.com/feder-observatory/stellarphot/issues?q=author%3AJuanCab" title="Bug reports">🐛</a> <a href="https://github.com/feder-observatory/stellarphot/commits?author=JuanCab" title="Code">💻</a> <a href="#design-JuanCab" title="Design">🎨</a> <a href="#ideas-JuanCab" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-JuanCab" title="Maintenance">🚧</a> <a href="https://github.com/feder-observatory/stellarphot/pulls?q=is%3Apr+reviewed-by%3AJuanCab" title="Reviewed Pull Requests">👀</a></td>
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

This project is Copyright (c) Matt Craig and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the [Astropy package template](https://github.com/astropy/package-template>)
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.
