# A package for transforming stellar photometry

[![Powered by Astropy Badge](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org) [![GitHub Workflow badge](https://github.com/feder-observatory/stellarphot/workflows/Test/badge.svg?branch=main)](https://github.com/feder-observatory/stellarphot/actions?query=workflow%3ATest)

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


## License

This project is Copyright (c) Matt Craig and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the [Astropy package template](https://github.com/astropy/package-template>)
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.


