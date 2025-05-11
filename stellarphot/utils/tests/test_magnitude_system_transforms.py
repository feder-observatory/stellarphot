# The goal here is to both test code and the data files
from pathlib import Path

import numpy as np
import pytest
from astropy.utils.data import get_pkg_data_path
from pydantic import ValidationError

from stellarphot.utils.magnitude_system_transforms import (
    MagnitudeSystem,
    MagnitudeSystemNames,
    MagnitudeSystemTransform,
    MagnitudeTransform,
    PanStarrs1ToJohnsonCousins,
)


def _check_json_roundtrip(model):
    """
    Check that the model can be serialized to JSON and then deserialized
    back to the same model.
    """
    json_str = model.model_dump_json()
    new_model = model.model_validate_json(json_str)
    assert model == new_model


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
        my_transform = MagnitudeTransform(
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
        my_mag_trans1 = MagnitudeTransform(
            name="test",
            from_passband="B",
            to_passband="gp1",
            polynomial_coefficients=fake_coeff,
            residual=0.1,
        )

        my_mag_trans2 = MagnitudeTransform(
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
            transform_coefficients={
                ("B", "gp1"): my_mag_trans1,
                ("V", "rp1"): my_mag_trans2,
            },
        )

        # Make sure we can access the transform coefficients as intended.
        assert my_transform.transform_coefficients[("B", "gp1")] == my_mag_trans1
        assert my_transform.transform_coefficients[("V", "rp1")] == my_mag_trans2

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
        assert ps1_to_jc.transform_coefficients[("gp1", "B")].polynomial == gp1_B_poly

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
