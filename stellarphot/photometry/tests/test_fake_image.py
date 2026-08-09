import numpy as np
import pytest

from stellarphot.photometry.tests.fake_image import (
    FakeCCDImage,
    make_gaussian_sources_image,
    make_moffat_sources_image,
)

SEED = 5432985


class TestMoffatSourcesImage:
    def test_moffat_source_has_requested_fwhm(self):
        # A rendered Moffat source should drop to half its peak value at a
        # radius of half the requested FWHM -- the definition of FWHM, and a
        # direct check of the gamma-from-FWHM conversion.
        from astropy.table import Table

        fwhm = 8.0
        # Center the source on a pixel so the peak is sampled exactly and
        # fwhm/2 lands exactly on another pixel center.
        source = Table({"x_mean": [50.0], "y_mean": [50.0], "amplitude": [100.0]})
        image = make_moffat_sources_image((101, 101), source, fwhm=fwhm)

        peak = image[50, 50]
        assert peak == pytest.approx(100.0, rel=1e-6)
        assert image[50, 50 + int(fwhm / 2)] == pytest.approx(peak / 2, rel=1e-6)

    def test_fake_ccd_image_moffat_fwhm(self):
        # The psf="moffat" path through FakeCCDImage should produce sources
        # whose measured FWHM is within ~20% of the requested value: the
        # measurement fits a Gaussian, which the Moffat's wings bias wide by
        # ~15% at alpha=2.5.
        from stellarphot.photometry import fast_fwhm_from_image

        fwhm = 7.0
        ccd = FakeCCDImage(seed=SEED, fwhm=fwhm, noise_dev=1.0, psf="moffat")
        measured = fast_fwhm_from_image(ccd, fwhm, noise=1.0, max_adu=40000)
        assert measured == pytest.approx(fwhm, rel=0.2)

    def test_gaussian_source_has_requested_fwhm(self):
        # Same half-max geometry check as the Moffat test above, for the
        # Gaussian renderer: the profile should drop to half its peak at a
        # radius of half the FWHM implied by the stddev columns.
        from astropy.stats import gaussian_fwhm_to_sigma
        from astropy.table import Table

        fwhm = 8.0
        sigma = fwhm * gaussian_fwhm_to_sigma
        # Center the source on a pixel so the peak is sampled exactly and
        # fwhm/2 lands exactly on another pixel center.
        source = Table(
            {
                "x_mean": [50.0],
                "y_mean": [50.0],
                "amplitude": [100.0],
                "x_stddev": [sigma],
                "y_stddev": [sigma],
                "theta": [0.0],
            }
        )
        image = make_gaussian_sources_image((101, 101), source)

        peak = image[50, 50]
        assert peak == pytest.approx(100.0, rel=1e-6)
        assert image[50, 50 + int(fwhm / 2)] == pytest.approx(peak / 2, rel=1e-6)

    def test_fake_ccd_image_rejects_unknown_psf(self):
        # The psf argument is validated at construction so a typo fails
        # fast with a clear error instead of silently falling back to a
        # Gaussian image.
        with pytest.raises(ValueError, match="psf"):
            FakeCCDImage(seed=SEED, fwhm=5.0, psf="airy")

    def test_fake_ccd_image_gaussian_unchanged(self):
        # The default psf stays Gaussian and the sources table keeps the
        # Gaussian column names all downstream consumers rely on.
        ccd = FakeCCDImage(seed=SEED, fwhm=5.0)
        for column in ("x_mean", "y_mean", "x_stddev", "y_stddev", "amplitude"):
            assert column in ccd.sources.colnames
        assert np.all(np.isfinite(ccd.data))
