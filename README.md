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


<<<<<<< HEAD
=======
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


>>>>>>> feder/main
## License

This project is Copyright (c) Matt Craig and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the [Astropy package template](https://github.com/astropy/package-template>)
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.
