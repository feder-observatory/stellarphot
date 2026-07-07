import astropy.units as u
import numpy as np
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
