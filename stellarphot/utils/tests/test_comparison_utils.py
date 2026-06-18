import astropy.units as u

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
