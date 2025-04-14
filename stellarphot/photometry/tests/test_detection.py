import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import CCDData
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import QTable
from astropy.utils.data import get_pkg_data_path
from astropy.utils.exceptions import AstropyUserWarning

from stellarphot import SourceListData
from stellarphot.photometry import compute_fwhm, source_detection
from stellarphot.photometry.tests.fake_image import FakeImage
from stellarphot.settings.models import FwhmMethods

# Make sure the tests are deterministic by using a random seed
SEED = 5432985


@pytest.mark.parametrize("units", [u.pixel, None])
def test_compute_fwhm(units):
    fake_image = FakeImage(seed=SEED)
    sources = fake_image.sources
    if units is not None:
        # It turns out having a unit on a column is not the same as
        # things in the column having units. The construct below ensures
        # that the source table values have units.
        # Do not try: sources['x_mean'] = sources['x_mean'] * units
        # Turns out individual values do NOT have units in that case.
        sources["x_mean"] = [v * units for v in sources["x_mean"]]
        sources["y_mean"] = [v * units for v in sources["y_mean"]]

    fwhm_x, fwhm_y = compute_fwhm(
        fake_image.image, sources, x_column="x_mean", y_column="y_mean"
    )

    expected_fwhm = np.array(sources["x_stddev"] * gaussian_sigma_to_fwhm)
    assert np.allclose(fwhm_x, expected_fwhm, rtol=1e-2)


@pytest.mark.parametrize("mask_by_nan", [True, False])
def test_compute_fwhm_with_missing_data(mask_by_nan):
    # Regression test for https://github.com/feder-observatory/stellarphot/issues/161
    # We should be able to find FWHM for a source even with NaNs in the image.
    fake_image = FakeImage(seed=SEED)
    sources = fake_image.sources
    x, y = sources["x_mean"].astype(int)[0], sources["y_mean"].astype(int)[0]
    image = fake_image.image.copy()

    if mask_by_nan:
        # Add a NaN to the image at the location of the first source. Note the
        # usual row/column swap when going to x/y coordinates.
        image[y, x] = np.nan
    else:
        image = CCDData(image, unit=u.adu, mask=np.zeros_like(image, dtype=bool))
        image.mask[y, x] = True

    # We expect a warning about NaNs in the image, so catch it
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=("Input data contains unmasked non-finite values "),
            category=AstropyUserWarning,
        )
    fwhm_x, fwhm_y = compute_fwhm(
        image, sources, x_column="x_mean", y_column="y_mean", fit_method=FwhmMethods.FIT
    )

    expected_fwhm = np.array(sources["x_stddev"] * gaussian_sigma_to_fwhm)
    assert np.allclose(fwhm_x, expected_fwhm, rtol=1e-2)


def test_detect_source_number_location():
    """
    Make sure we detect the sources in the input table....
    """
    fake_image = FakeImage(seed=SEED)
    sources = QTable(
        fake_image.sources,
        units={
            "x_mean": u.pixel,
            "y_mean": u.pixel,
            "x_stddev": u.pixel,
            "y_stddev": u.pixel,
        },
    )
    # print(sources)
    # Pass only one value for the sky background for source detection
    sky_per_pix = sources["sky_per_pix_avg"].mean()
    found_sources = source_detection(
        fake_image.image,
        fwhm=2 * sources["x_stddev"].mean(),
        threshold=10,
        sky_per_pix_avg=sky_per_pix,
    )
    # Sort by flux so we can reliably match them
    sources.sort("amplitude")
    found_sources.sort("flux")

    # Do we have the right number of sources?
    assert len(sources) == len(found_sources)

    for inp, out in zip(sources, found_sources, strict=True):
        # Do the positions match?
        np.testing.assert_allclose(out["xcenter"], inp["x_mean"], rtol=1e-5, atol=0.05)
        np.testing.assert_allclose(out["ycenter"], inp["y_mean"], rtol=1e-5, atol=0.05)
        np.testing.assert_allclose(
            gaussian_sigma_to_fwhm * (inp["x_stddev"] + inp["y_stddev"]) / 2,
            out["width"],
            rtol=1e-5,
            atol=0.05,
        )


def test_detect_source_with_padding():
    """
    Make sure we detect the sources in the input table....
    """
    fake_image = FakeImage(seed=SEED)
    sources = QTable(
        fake_image.sources,
        units={
            "x_mean": u.pixel,
            "y_mean": u.pixel,
            "x_stddev": u.pixel,
            "y_stddev": u.pixel,
        },
    )
    # Pass only one value for the sky background for source detection
    sky_per_pix = sources["sky_per_pix_avg"].mean()
    # Padding was chosen to be large enough to ensure that one of the sources in
    # test_sources.csv would land too close to the edge of the image.
    found_sources = source_detection(
        fake_image.image,
        fwhm=2 * sources["x_stddev"].mean(),
        threshold=10,
        sky_per_pix_avg=sky_per_pix,
        padding=95,
    )

    # Did we drop one source because it was too close to the edge?
    assert len(sources) - 1 == len(found_sources)


@pytest.mark.parametrize(
    "fit_method", [FwhmMethods.FIT, FwhmMethods.PROFILE, FwhmMethods.MOMENTS]
)
def test_fwhm_computation(fit_method):
    # Regression test for #490, in which FWHM computation is incorrect
    # because the image hasn't been background subtracted.
    ccd_file = get_pkg_data_path("data/cutout_for_fwhm_test.fits")
    source_list_file = get_pkg_data_path("data/source_list_for_fwhm_test.ecsv")

    ccd = CCDData.read(ccd_file, unit=u.adu)
    source_list = SourceListData.read(source_list_file)
    # Value below is from the image the cutout was taken from
    source_list["sky_per_pix_avg"] = np.array([83.69])

    fwhm_x, fwhm_y = compute_fwhm(
        ccd,
        source_list,
        fwhm_estimate=6.85179,
        x_column="xcenter",
        y_column="ycenter",
        sky_per_pix_column="sky_per_pix_avg",
        fit_method=fit_method,
    )

    avg_fwhm = np.mean([fwhm_x, fwhm_y])
    if fit_method == FwhmMethods.MOMENTS:
        assert avg_fwhm > 7
    else:
        assert np.isclose(avg_fwhm, 6.6, rtol=0.1)


def test_compute_fwhm_input_options():
    with pytest.raises(ValueError, match="Cannot specify both "):
        compute_fwhm(
            None, None, sky_per_pix_avg=10, sky_per_pix_column="sky_per_pix_avg"
        )

    # Make a table with a single column
    table = QTable({"sky_per_pix_avg": [10]})
    spp_column = "foo"
    with pytest.raises(
        ValueError, match=f"Column {spp_column} not found in sources table"
    ):
        compute_fwhm(None, table, sky_per_pix_column=spp_column)

    fake_image = FakeImage(seed=SEED)
    sources = fake_image.sources

    fit_method = "foo"
    with pytest.raises(ValueError, match=f"Unknown fit method: {fit_method}"):
        compute_fwhm(
            fake_image.image,
            sources,
            fit_method=fit_method,
            x_column="x_mean",
            y_column="y_mean",
        )
