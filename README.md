# hot
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)

Applying iterative sine fitting, Oxford CBV correction and BLS search to detect transits and eclipses of hot pulsating stars in the nominal Kepler mission.

## Installation Instructions

First

	`pip install astropy fitsio lightkurve acor`

Not all of the dependencies of hot can be installed via pip. You need to unzip the `kplr_cbv` in ../data/, which is the (large) [Kepler CBV file](http://archive.stsci.edu/kepler/cbv.html).

You then need to separately clone and install [OxKeplerSC](https://github.com/OxES/OxKeplerSC), [PyBLS](https://github.com/hpparvi/PyBLS), [k2ps](https://github.com/hpparvi/k2ps), [PyTransit](https://github.com/hpparvi/PyTransit), [PyExoTk](https://github.com/hpparvi/PyExoTK)