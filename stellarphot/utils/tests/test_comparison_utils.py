import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.wcs import WCS

from stellarphot.utils import comparison_utils


class _FakeCCD:
    """Minimal stand-in for a CCDData that only needs a ``wcs`` attribute."""

    wcs = object()


def test_set_up_passes_magnitude_limit_to_vsx(monkeypatch):
    # Regression test for #43. set_up should forward a magnitude limit to the
    # VSX lookup so the comparison viewer can apply the same dim-magnitude
    # cutoff to variable stars that it uses for comparison stars.
    captured = {}

    def fake_vsx_vizier(wcs, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        # Behave like "no variables found" so set_up returns without needing a
        # real query result.
        raise RuntimeError("no VSX results")

    monkeypatch.setattr(comparison_utils, "vsx_vizier", fake_vsx_vizier)

    result = comparison_utils.set_up(_FakeCCD(), magnitude_limit=13.5)

    assert captured["magnitude_limit"] == 13.5
    assert result == []


def test_set_up_defaults_to_no_magnitude_limit(monkeypatch):
    # By default no magnitude limit is applied to the VSX lookup.
    captured = {}

    def fake_vsx_vizier(wcs, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        raise RuntimeError("no VSX results")

    monkeypatch.setattr(comparison_utils, "vsx_vizier", fake_vsx_vizier)

    comparison_utils.set_up(_FakeCCD())

    assert captured["magnitude_limit"] is None
    # The search radius is still passed through unchanged.
    assert captured["radius"] == 0.5 * u.degree


def _non_square_ccd(shape=(200, 400)):
    # numpy shape is (ny, nx), so this image is 400 pixels wide (x) and
    # 200 pixels tall (y).
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    wcs.wcs.cdelt = [-2.0e-4, 2.0e-4]
    wcs.wcs.crval = [30.0, 45.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return CCDData(np.zeros(shape), wcs=wcs, unit="adu")


def test_in_field_non_square_image():
    # Regression test for #589. in_field unpacked the numpy image shape as
    # (nx, ny), swapping the x and y bounds on non-square images. That
    # excluded valid stars near the long edge and included off-image stars.
    ccd = _non_square_ccd()

    # Pixel positions (x, y) of the test stars:
    #   0: inside the image, but excluded by the buggy bounds (x > 200)
    #   1: outside the image (y > 200), but included by the buggy bounds
    #   2: inside by either version of the bounds
    #   3: outside by either version of the bounds
    xs = np.array([300.0, 100.0, 50.0, 500.0])
    ys = np.array([100.0, 300.0, 50.0, 500.0])
    coords = ccd.wcs.pixel_to_world(xs, ys)

    apass = Table({"id": np.arange(len(xs)), "coords": coords})
    good_stars = np.ones(len(apass), dtype=bool)

    ent = comparison_utils.in_field(apass["coords"], ccd, apass, good_stars)

    assert sorted(ent["id"]) == [0, 2]


def test_crossmatch_apass_to_vsx_and_targets_reports_separations(mocker):
    # crossmatch_APASS2VSX should report the on-sky separation between each
    # APASS star and its nearest match in the VSX table and in the
    # target-list (RD) table.
    apass_raw = Table({"ra": [10.0], "dec": [20.0]})
    mocker.patch.object(comparison_utils, "apass_dr9", return_value=apass_raw)

    # VSX star offset from the APASS star by exactly 2 arcsec in declination.
    vsx = Table({"coords": SkyCoord(ra=[10.0] * u.deg, dec=[20.0 + 2 / 3600] * u.deg)})
    # Target-list star offset by exactly 5 arcsec in declination.
    RD = Table({"coords": SkyCoord(ra=[10.0] * u.deg, dec=[20.0 + 5 / 3600] * u.deg)})

    apass, v_angle, RD_angle = comparison_utils.crossmatch_APASS2VSX(
        _FakeCCD(), RD, vsx
    )

    # The mocked apass_dr9 return value is used as-is (with a coords column
    # added), so the coordinates round trip.
    assert apass is apass_raw
    np.testing.assert_allclose(v_angle.arcsec, [2.0], atol=1e-6)
    np.testing.assert_allclose(RD_angle.arcsec, [5.0], atol=1e-6)


def test_crossmatch_apass_handles_empty_vsx_list(mocker):
    # set_up returns a plain empty list, not a Table, when no VSX variables
    # are found (see set_up above). crossmatch_APASS2VSX must handle that
    # falsy-but-not-a-Table value without erroring, leaving v_angle empty
    # while still matching against a non-empty target list.
    apass_raw = Table({"ra": [10.0], "dec": [20.0]})
    mocker.patch.object(comparison_utils, "apass_dr9", return_value=apass_raw)

    RD = Table({"coords": SkyCoord(ra=[10.0] * u.deg, dec=[20.0 + 5 / 3600] * u.deg)})

    apass, v_angle, RD_angle = comparison_utils.crossmatch_APASS2VSX(_FakeCCD(), RD, [])

    assert v_angle == []
    np.testing.assert_allclose(RD_angle.arcsec, [5.0], atol=1e-6)


def test_mag_scale_selects_passband():
    # Only stars in the requested passband should be considered, regardless
    # of how well their magnitude and position would otherwise qualify.
    coords = SkyCoord(ra=[10, 11] * u.deg, dec=[20, 20] * u.deg)
    apass = Table({"passband": ["SR", "SG"], "mag": [10.0, 10.0], "coords": coords})
    far = u.Quantity([100, 100], u.arcsec)

    _, good_stars = comparison_utils.mag_scale(10.0, apass, far, far, passband="SR")
    np.testing.assert_array_equal(good_stars, [True, False])

    _, good_stars = comparison_utils.mag_scale(10.0, apass, far, far, passband="SG")
    np.testing.assert_array_equal(good_stars, [False, True])


def test_mag_scale_brighter_and_dimmer_magnitude_cuts():
    # brighter_dmag/dimmer_dmag bound how far a comparison star's magnitude
    # may be from the target's; stars outside that window are excluded even
    # though they are the right passband and not close to anything else.
    cmag = 10.0
    brighter_dmag = 0.44
    dimmer_dmag = 0.75
    coords = SkyCoord(ra=[10, 11, 12, 13] * u.deg, dec=[20] * 4 * u.deg)
    apass = Table(
        {
            "passband": ["SR"] * 4,
            # too bright, just inside the bright cut, just inside the dim
            # cut, too dim.
            "mag": [
                cmag - brighter_dmag - 0.01,
                cmag - brighter_dmag + 0.01,
                cmag + dimmer_dmag - 0.01,
                cmag + dimmer_dmag + 0.01,
            ],
            "coords": coords,
        }
    )
    far = u.Quantity([100] * 4, u.arcsec)

    _, good_stars = comparison_utils.mag_scale(
        cmag, apass, far, far, brighter_dmag=brighter_dmag, dimmer_dmag=dimmer_dmag
    )
    np.testing.assert_array_equal(good_stars, [False, True, True, False])


def test_mag_scale_excludes_stars_too_close_to_vsx_or_targets():
    # Comparison stars within 2 arcsec of a VSX variable or a target-list
    # star are excluded, even when passband and magnitude are otherwise fine.
    coords = SkyCoord(ra=[10, 11, 12] * u.deg, dec=[20] * 3 * u.deg)
    apass = Table({"passband": ["SR"] * 3, "mag": [10.0] * 3, "coords": coords})

    # Star 0 is just inside the VSX exclusion radius, star 1 is just inside
    # the target-list exclusion radius, star 2 is just outside both.
    v_angle = u.Quantity([1.9, 100, 2.1], u.arcsec)
    RD_angle = u.Quantity([100, 1.9, 2.1], u.arcsec)

    apass_good_coord, good_stars = comparison_utils.mag_scale(
        10.0, apass, v_angle, RD_angle
    )

    np.testing.assert_array_equal(good_stars, [False, False, True])
    assert len(apass_good_coord) == 1
