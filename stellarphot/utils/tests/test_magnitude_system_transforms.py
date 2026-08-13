# The goal here is to both test code and the data files
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils.data import get_pkg_data_path
from pydantic import ValidationError

from stellarphot import apass_dr9, refcat2
from stellarphot.utils.magnitude_system_transforms import (
    MagnitudeSystem,
    MagnitudeSystemNames,
    MagnitudeSystemTransform,
    MagnitudeTransformPolynomial,
    PanStarrs1ToJohnsonCousins,
    USNOPrimeToSDSSDR7,
    transform_apass_bands,
    transform_refcat2_bands,
)


def _check_json_roundtrip(model):
    """
    Check that the model can be serialized to JSON and then deserialized
    back to the same model.
    """
    json_str = model.model_dump_json()
    new_model = model.model_validate_json(json_str)
    assert model == new_model


def _passband_table(bands, mags, errors=None):
    """
    Build a table shaped like the output of ``CatalogData.passband_columns``:
    one ``mag_<band>`` column per band and, when ``errors`` is given, a
    matching ``mag_error_<band>`` column.
    """
    table = Table()
    for band, values in zip(bands, mags, strict=True):
        table[f"mag_{band}"] = np.asarray(values, dtype=float)
    if errors is not None:
        for band, values in zip(bands, errors, strict=True):
            table[f"mag_error_{band}"] = np.asarray(values, dtype=float)
    return table


# The Jester et al. (2005) transformations used for APASS, from
# http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php -- R and I are
# compositions of the tabulated relations (R = V - (V-R), I = R - (Rc-Ic)),
# so their rms residuals combine in quadrature: V and V-R for R, plus Rc-Ic
# for I.
_JESTER_R_COEFFS = (0.41, -0.5, 1.09)
_JESTER_I_COEFFS = (0.41, -1.5, 2.09)
_JESTER_R_RMS = np.sqrt(0.01**2 + 0.03**2)
_JESTER_I_RMS = np.sqrt(0.01**2 + 0.03**2 + 0.01**2)


class TestMagnitudeSystem:
    def test_can_create_and_jc_file_is_good(self):
        jc_bands = ["U", "B", "V", "Rc", "Ic"]
        sys = MagnitudeSystem(name=MagnitudeSystemNames.JC, passbands=jc_bands)
        assert sys.name == MagnitudeSystemNames.JC
        assert sys.passbands == jc_bands

        # Check that Johnson-Cousins that we ship is as expected
        from_data = Path(
            get_pkg_data_path("data/JohnsonCousins.json", package="stellarphot.utils")
        )
        assert from_data.exists()
        assert sys == MagnitudeSystem.model_validate_json(from_data.read_text())

        # Check serialization roundtrip
        _check_json_roundtrip(sys)

    def test_no_unknown_systems(self):
        with pytest.raises(ValidationError):
            MagnitudeSystem(name="Unknown", passbands=["U", "B", "V", "R", "I"])

    def test_panstarrs1_system_def(self):
        ps1_file = Path(
            get_pkg_data_path("data/pan-starrs1.json", package="stellarphot.utils")
        )
        ps1 = MagnitudeSystem.model_validate_json(ps1_file.read_text())
        assert ps1.name == MagnitudeSystemNames.PANSTARRS1
        assert ps1.passbands == ["gp1", "rp1", "ip1", "zp1", "yp1", "wp1"]


class TestMagnitudeTransform:
    def test_make_transform(self):
        fake_coeff = [0.1, 0.2, 0.3]
        # This is NOT a real transform....just making sure that
        # the transform can be made
        my_transform = MagnitudeTransformPolynomial(
            name="test",
            from_passband="B",
            to_passband="gp1",
            polynomial_coefficients=fake_coeff,
            residual=0.1,
        )

        assert my_transform.name == "test"
        assert np.polynomial.Polynomial(fake_coeff) == my_transform.polynomial
        val = 2
        assert np.isclose(
            my_transform.polynomial(val), (0.1 + 0.2 * val + 0.3 * val**2)
        )

        # check roundtrip serialization
        _check_json_roundtrip(my_transform)


class TestMagnitudeSystemTransform:
    def test_make_transform(self):
        my_jc_system = MagnitudeSystem(
            name=MagnitudeSystemNames.JC, passbands=["B", "V"]
        )
        my_ps1_system = MagnitudeSystem(
            name=MagnitudeSystemNames.PANSTARRS1, passbands=["gp1", "rp1"]
        )

        fake_coeff = [0.1, 0.2, 0.3]
        my_mag_trans1 = MagnitudeTransformPolynomial(
            name="test",
            from_passband="B",
            to_passband="gp1",
            polynomial_coefficients=fake_coeff,
            residual=0.1,
        )

        my_mag_trans2 = MagnitudeTransformPolynomial(
            name="test",
            from_passband="V",
            to_passband="rp1",
            polynomial_coefficients=fake_coeff,
            residual=0.1,
        )

        my_transform = MagnitudeSystemTransform(
            name="test",
            reference="some paper I will never write",
            from_system=my_jc_system,
            to_system=my_ps1_system,
            transform_information={
                ("B", "gp1"): my_mag_trans1,
                ("V", "rp1"): my_mag_trans2,
            },
        )

        # Make sure we can access the transform coefficients as intended.
        assert my_transform.transform_information[("B", "gp1")] == my_mag_trans1
        assert my_transform.transform_information[("V", "rp1")] == my_mag_trans2

        # Serialize the model and make sure that the transform keys appear
        # as expected.
        serialized = my_transform.model_dump_json()
        assert "B,gp1" in serialized
        assert "V,rp1" in serialized

        # Check that the transform can roundtrip to json
        _check_json_roundtrip(my_transform)


class TestPanStarrs1ToJohnsonCousins:
    @pytest.mark.parametrize("setup_method", ["class", "load"])
    def test_transform(self, setup_method):
        if setup_method == "class":
            # Here we just read in the transform on disk rather than doing it manually.
            ps1_file = Path(
                get_pkg_data_path("data/PS1_to_JC.json", package="stellarphot.utils")
            )
            ps1_to_jc = PanStarrs1ToJohnsonCousins.model_validate_json(
                ps1_file.read_text()
            )
        else:
            ps1_to_jc = PanStarrs1ToJohnsonCousins.load()

        # Spot check a couple of transform relationships
        # Expected polynomials from the paper
        gp1_B_poly = np.polynomial.Polynomial([0.212, 0.556, 0.034])
        assert ps1_to_jc.transform_information[("gp1", "B")].polynomial == gp1_B_poly

        # Check that the transform works
        fake_gp1_mags = np.array(
            [
                [20.0, 19.0, 18.0, 17.0, 16.0, 15.0],
                [20.0, 19.0, 18.0, 17.0, 16.0, 15.0],
                [20.0, 19.0, 18.0, 17.0, 16.0, 15.0],
            ]
        )
        # Transform just a single row
        fake_jc_mags = ps1_to_jc(fake_gp1_mags[1, :])
        assert fake_jc_mags.shape == (4,)
        assert np.isclose(
            fake_jc_mags[0],
            gp1_B_poly(fake_gp1_mags[0, 0] - fake_gp1_mags[0, 1]) + fake_gp1_mags[0, 0],
        )

        # Transform it all
        fake_jc_mags = ps1_to_jc(fake_gp1_mags)
        assert fake_jc_mags.shape == (3, 4)
        assert np.isclose(
            fake_jc_mags[0, 0],
            gp1_B_poly(fake_gp1_mags[0, 0] - fake_gp1_mags[0, 1]) + fake_gp1_mags[0, 0],
        )

    def test_transform_color_uses_band_lookup_not_position(self):
        # The gp1 - rp1 color used internally must be found by looking up
        # the band names in from_system.passbands, not by assuming gp1 and
        # rp1 are the first two columns of the input array. Reordering the
        # passbands (and the input magnitudes to match) should not change
        # the result.
        ps1_to_jc = PanStarrs1ToJohnsonCousins.load()
        band_values = {
            "gp1": 20.0,
            "rp1": 19.0,
            "ip1": 18.0,
            "zp1": 17.0,
            "yp1": 16.0,
            "wp1": 15.0,
        }
        original_order = ps1_to_jc.from_system.passbands
        original_mags = np.array([band_values[b] for b in original_order])
        expected = ps1_to_jc(original_mags)

        # Reverse the passband order so that gp1/rp1 are no longer the
        # first two entries, and reorder the magnitudes to match.
        reordered_order = list(reversed(original_order))
        assert reordered_order != original_order
        reordered_system = ps1_to_jc.from_system.model_copy(
            update={"passbands": reordered_order}
        )
        ps1_to_jc_reordered = ps1_to_jc.model_copy(
            update={"from_system": reordered_system}
        )
        reordered_mags = np.array([band_values[b] for b in reordered_order])

        result = ps1_to_jc_reordered(reordered_mags)
        assert np.allclose(result, expected)

    def test_transform_missing_required_band_raises(self):
        # If the input passbands don't include gp1 or rp1, the transform
        # cannot compute the g-r color it needs and should raise a clear
        # error rather than silently using the wrong columns.
        ps1_to_jc = PanStarrs1ToJohnsonCousins.load()
        passbands_without_rp1 = [
            band for band in ps1_to_jc.from_system.passbands if band != "rp1"
        ]
        missing_system = ps1_to_jc.from_system.model_copy(
            update={"passbands": passbands_without_rp1}
        )
        ps1_to_jc_missing_rp1 = ps1_to_jc.model_copy(
            update={"from_system": missing_system}
        )

        with pytest.raises(ValueError, match="rp1"):
            ps1_to_jc_missing_rp1(np.zeros(len(passbands_without_rp1)))

    def test_transform_zero_errors_propagate_to_residual(self):
        # With zero measurement error the propagated error is exactly the
        # published residual of each transformation -- the noise floor of
        # the fit itself.
        ps1_to_jc = PanStarrs1ToJohnsonCousins.load()
        mags = np.array([[20.0, 19.5, 19.0, 18.5, 18.0, 17.5]] * 2)
        jc_mags, jc_errors = ps1_to_jc(mags, from_magnitude_errors=np.zeros_like(mags))
        expected_residuals = [
            ps1_to_jc.transform_information[bands].residual
            for bands in (("gp1", "B"), ("gp1", "V"), ("rp1", "Rc"), ("ip1", "Ic"))
        ]
        assert jc_errors.shape == (2, 4)
        assert np.allclose(jc_errors, expected_residuals)
        # Asking for errors must not change the magnitudes themselves
        assert np.allclose(jc_mags, ps1_to_jc(mags))

    def test_transform_errors_match_finite_differences(self):
        # The propagated error should agree with numerically estimated
        # partial derivatives of the magnitude transform, combined with the
        # input errors and the published residual in quadrature.
        ps1_to_jc = PanStarrs1ToJohnsonCousins.load()
        mags = np.array([20.0, 19.5, 19.0, 18.5, 18.0, 17.5])
        errors = np.array([0.01, 0.02, 0.03, 0.04, 0.0, 0.0])
        jc_mags, jc_errors = ps1_to_jc(mags, from_magnitude_errors=errors)
        assert jc_mags.shape == (4,)
        assert jc_errors.shape == (4,)

        delta = 1e-6
        expected = np.zeros(4)
        for out_index, bands in enumerate(
            (("gp1", "B"), ("gp1", "V"), ("rp1", "Rc"), ("ip1", "Ic"))
        ):
            variance = ps1_to_jc.transform_information[bands].residual ** 2
            for in_index in range(len(mags)):
                up, down = mags.copy(), mags.copy()
                up[in_index] += delta
                down[in_index] -= delta
                partial = (ps1_to_jc(up) - ps1_to_jc(down))[out_index] / (2 * delta)
                variance += (partial * errors[in_index]) ** 2
            expected[out_index] = np.sqrt(variance)
        assert np.allclose(jc_errors, expected)


class TestCatalogTransforms:
    @pytest.mark.remote_data
    def test_apass_adds_RI(self):
        # Get some APASS data
        apass = apass_dr9(SkyCoord(0, 0, unit="degree"), radius="1 arcmin")
        assert "mag_R" not in apass.columns
        assert "mag_I" not in apass.columns

        # Transform the data to add R and I
        apass_trans = apass.passband_columns(
            ["R", "I"], transformer=transform_apass_bands
        )
        assert "mag_R" in apass_trans.columns
        assert "mag_I" in apass_trans.columns
        # The catalog errors must be propagated into the transformed bands
        assert "mag_error_R" in apass_trans.columns
        assert "mag_error_I" in apass_trans.columns

    @pytest.mark.remote_data
    def test_apass_can_apply_usno(self):
        # Get some APASS data
        apass = apass_dr9(SkyCoord(0, 0, unit="degree"), radius="1 arcmin")
        assert "mag_R" not in apass.columns
        assert "mag_I" not in apass.columns

        # Transform the data to add R and I
        apass_trans = apass.passband_columns(
            ["R", "I"],
            transformer=transform_apass_bands,
            transformer_kwargs=dict(apply_sdssdr7_transform=True),
        )
        assert "mag_R" in apass_trans.columns
        assert "mag_I" in apass_trans.columns
        assert "mag_error_R" in apass_trans.columns
        assert "mag_error_I" in apass_trans.columns

    @pytest.mark.remote_data
    def test_refcats_adds_BVRI(self):
        # Get some APASS data
        refcat = refcat2(SkyCoord(0, 0, unit="degree"), radius="1 arcmin")
        assert "mag_B" not in refcat.columns
        assert "mag_V" not in refcat.columns
        assert "mag_R" not in refcat.columns
        assert "mag_I" not in refcat.columns

        # Transform the data to add B, V, R, and I
        refcat_trans = refcat.passband_columns(
            ["B", "V", "R", "I"], transformer=transform_refcat2_bands
        )
        assert "mag_B" in refcat_trans.columns
        assert "mag_V" in refcat_trans.columns
        assert "mag_R" in refcat_trans.columns
        assert "mag_I" in refcat_trans.columns
        for band in ("B", "V", "R", "I"):
            assert f"mag_error_{band}" in refcat_trans.columns

        # Make sure an error is raised when a unknown band is requested
        with pytest.raises(
            ValueError, match="Transformer did not add columns for passbands"
        ):
            refcat.passband_columns(["X"], transformer=transform_refcat2_bands)

    def test_apass_transform_propagates_errors(self):
        # The Jester transforms are linear, R = 0.41 g - 0.5 r + 1.09 i - 0.23
        # and I = 0.41 g - 1.5 r + 2.09 i - 0.44, so the propagated error is
        # the coefficient-weighted quadrature sum of the native errors plus
        # the rms residual of the transform itself.
        eg = np.array([0.01, 0.04])
        er = np.array([0.02, 0.05])
        ei = np.array([0.03, 0.06])
        table = _passband_table(
            ["SG", "SR", "SI"],
            [[14.0, 15.0], [13.6, 14.4], [13.4, 14.1]],
            errors=[eg, er, ei],
        )
        transformed = transform_apass_bands(table)

        for band, coeffs, rms in (
            ("R", _JESTER_R_COEFFS, _JESTER_R_RMS),
            ("I", _JESTER_I_COEFFS, _JESTER_I_RMS),
        ):
            cg, cr, ci = coeffs
            expected = np.sqrt(
                (cg * eg) ** 2 + (cr * er) ** 2 + (ci * ei) ** 2 + rms**2
            )
            assert np.allclose(transformed[f"mag_error_{band}"], expected)
            assert np.allclose(
                transformed[f"mag_error_{band}C"], transformed[f"mag_error_{band}"]
            )

    def test_apass_transform_without_errors_adds_no_error_columns(self):
        # A table without native error columns still gets magnitudes, and
        # no error columns are invented for it.
        table = _passband_table(["SG", "SR", "SI"], [[14.0], [13.6], [13.4]])
        transformed = transform_apass_bands(table)
        assert "mag_R" in transformed.colnames
        assert "mag_I" in transformed.colnames
        assert "mag_error_R" not in transformed.colnames
        assert "mag_error_I" not in transformed.colnames

    def test_apass_usno_transform_propagates_errors(self):
        # With the USNO->SDSS matrix applied first the transform is still
        # linear, so the propagated error must match finite-difference
        # partials of the full magnitude transform combined with the native
        # errors, plus the Jester rms in quadrature.
        native_bands = ["SG", "SR", "SI"]
        errors = [
            np.array([0.01, 0.04]),
            np.array([0.02, 0.05]),
            np.array([0.03, 0.06]),
        ]
        table = _passband_table(
            native_bands, [[14.0, 15.0], [13.6, 14.4], [13.4, 14.1]], errors=errors
        )
        transformed = transform_apass_bands(table.copy(), apply_sdssdr7_transform=True)

        delta = 1e-6
        for band, rms in (("R", _JESTER_R_RMS), ("I", _JESTER_I_RMS)):
            variance = np.full(2, rms**2)
            for native, err in zip(native_bands, errors, strict=True):
                up, down = table.copy(), table.copy()
                up[f"mag_{native}"] = up[f"mag_{native}"] + delta
                down[f"mag_{native}"] = down[f"mag_{native}"] - delta
                partial = (
                    np.asarray(
                        transform_apass_bands(up, apply_sdssdr7_transform=True)[
                            f"mag_{band}"
                        ]
                    )
                    - np.asarray(
                        transform_apass_bands(down, apply_sdssdr7_transform=True)[
                            f"mag_{band}"
                        ]
                    )
                ) / (2 * delta)
                variance += (partial * err) ** 2
            assert np.allclose(transformed[f"mag_error_{band}"], np.sqrt(variance))

    def test_refcat2_transform_propagates_errors(self):
        # Each Johnson-Cousins band is native + P(g - r), so the partials
        # are P'(c) through the color plus one for the native band, and the
        # published residual of each fit adds in quadrature. The polynomials
        # and residuals here are written out from the Pan-STARRS1 paper
        # independently of the shipped JSON file.
        eg = np.array([0.01, 0.04])
        er = np.array([0.02, 0.05])
        ei = np.array([0.03, 0.06])
        ez = np.array([0.04, 0.07])
        table = _passband_table(
            ["SG", "SR", "SI", "SZ"],
            [[14.0, 15.0], [13.6, 14.4], [13.4, 14.1], [13.3, 14.0]],
            errors=[eg, er, ei, ez],
        )
        transformed = transform_refcat2_bands(table)

        color = np.asarray(table["mag_SG"]) - np.asarray(table["mag_SR"])
        published = {
            "B": (np.polynomial.Polynomial([0.212, 0.556, 0.034]), 0.032, "g"),
            "V": (np.polynomial.Polynomial([0.005, -0.536, 0.011]), 0.012, "g"),
            "R": (np.polynomial.Polynomial([-0.137, -0.108, -0.029]), 0.015, "r"),
            "I": (np.polynomial.Polynomial([-0.366, -0.136, -0.018]), 0.017, "i"),
        }
        for band, (poly, residual, native) in published.items():
            dpdc = poly.deriv()(color)
            partial_g = dpdc + (native == "g")
            partial_r = -dpdc + (native == "r")
            partial_i = 1.0 * (native == "i")
            # The SZ error must not appear: no transform uses the z band.
            expected = np.sqrt(
                (partial_g * eg) ** 2
                + (partial_r * er) ** 2
                + (partial_i * ei) ** 2
                + residual**2
            )
            assert np.allclose(transformed[f"mag_error_{band}"], expected)
        assert np.allclose(transformed["mag_error_RC"], transformed["mag_error_R"])
        assert np.allclose(transformed["mag_error_IC"], transformed["mag_error_I"])

    def test_refcat2_transform_without_errors_adds_no_error_columns(self):
        table = _passband_table(
            ["SG", "SR", "SI", "SZ"], [[14.0], [13.6], [13.4], [13.3]]
        )
        transformed = transform_refcat2_bands(table)
        for band in ("B", "V", "R", "I"):
            assert f"mag_{band}" in transformed.colnames
            assert f"mag_error_{band}" not in transformed.colnames


class TestUSNOPrimeToSDSSDR7:
    # The reference for the transform is
    # https://classic.sdss.org/dr7/algorithms/jeg_photometric_eq_dr1.php#usno2SDSS
    # which gives, with all zpOffsets zero,
    #
    #   u = u'
    #   g = g' + 0.060 * ((g' - r') - 0.53)
    #   r = r' + 0.035 * ((r' - i') - 0.21)
    #   i = i' + 0.041 * ((r' - i') - 0.21)
    #   z = z' - 0.030 * ((i' - z') - 0.09)

    def test_transform_zeropoint_colors_are_fixed_point(self):
        # A star whose colors are exactly the zeropoint colors of the transform
        # ((u'-g') = 1.39, (g'-r') = 0.53, (r'-i') = 0.21, (i'-z') = 0.09) has
        # every color correction vanish, so its ugriz magnitudes must equal its
        # u'g'r'i'z' magnitudes. Unlike an all-zeros input, this exercises the
        # diagonal and off-diagonal color coefficients together.
        z_p = 10.0
        i_p = z_p + 0.09
        r_p = i_p + 0.21
        g_p = r_p + 0.53
        u_p = g_p + 1.39
        inp_mag = np.asarray([u_p, g_p, r_p, i_p, z_p, 1.0])
        usno_to_sdss = USNOPrimeToSDSSDR7.load()
        out_mag = usno_to_sdss(inp_mag)
        assert np.allclose(out_mag[:5], inp_mag[:5], atol=1e-6, rtol=0)

    def test_transform_matches_reference_equations(self):
        # Check every coefficient of the matrix against the published
        # equations, using colors that are *not* the zeropoint colors so that
        # each color term contributes. Two stars at once also checks that the
        # matrix multiplication handles a (n_bands + 1, n_stars) input.
        u_p = np.asarray([15.2, 13.1])
        g_p = np.asarray([14.7, 12.8])
        r_p = np.asarray([13.9, 12.6])
        i_p = np.asarray([13.4, 12.5])
        z_p = np.asarray([13.1, 12.45])
        inp_mag = np.asarray([u_p, g_p, r_p, i_p, z_p, np.ones_like(u_p)])

        expected = np.asarray(
            [
                u_p,
                g_p + 0.060 * ((g_p - r_p) - 0.53),
                r_p + 0.035 * ((r_p - i_p) - 0.21),
                i_p + 0.041 * ((r_p - i_p) - 0.21),
                z_p - 0.030 * ((i_p - z_p) - 0.09),
                np.ones_like(u_p),
            ]
        )
        usno_to_sdss = USNOPrimeToSDSSDR7.load()
        out_mag = usno_to_sdss(inp_mag)
        assert np.allclose(out_mag, expected, atol=1e-6, rtol=0)

    def test_transform_wrong_shape_raises(self):
        # Input without the constant-term row should raise, not silently
        # produce wrong magnitudes.
        usno_to_sdss = USNOPrimeToSDSSDR7.load()
        with pytest.raises(
            ValueError,
            match=r"expected 5 passband rows plus a final row of ones \(6 rows total\)",
        ):
            usno_to_sdss(np.zeros(5))
