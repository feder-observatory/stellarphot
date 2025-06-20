from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from astropy.utils.data import get_pkg_data_path
from pydantic import BaseModel, BeforeValidator, Field

__all__ = [
    "MagnitudeSystemNames",
    "MagnitudeSystem",
    "MagnitudeTransformPolynomial",
    "MagnitudeSystemTransform",
    "PanStarrs1ToJohnsonCousins",
]


class MagnitudeSystemNames(StrEnum):
    """Enum for different magnitude systems."""

    AB = "AB"
    Vega = "Vega"
    JC = "JC"
    SDSSDR7 = "SDSSDR7"
    USNO_SDSS = "USNO_SDSS"
    GAIA = "GAIA"
    TESS = "TESS"
    PANSTARRS1 = "PANSTARRS1"
    UGRIZ = "ugriz"


# class MagnitudePassband(BaseModel):
#     name: str
#     # passband_range: tuple[float, float]
#     # Down the road add something like symphot.SpectralElement


class MagnitudeSystem(BaseModel):
    """
    Class for a magnitude system.
    """

    name: MagnitudeSystemNames
    passbands: Annotated[
        list[str],
        Field(description=("List of passbands in the system.")),
    ]


class MagnitudeTransformPolynomial(BaseModel):
    """
    Class for magnitude transformation from one set of passbands to another that is best
    represented by a polynomial.
    """

    name: str
    from_passband: Annotated[
        str,
        Field(description="Passband to transform from."),
    ]
    to_passband: Annotated[
        str,
        Field(description="Passband to transform to."),
    ]
    polynomial_coefficients: Annotated[
        list[float],
        Field(
            description=(
                "Coefficients for the transformation. The "
                "number of coefficients depends on the transformation."
            )
        ),
    ]
    residual: Annotated[
        float,
        Field(description=("Residual of the transformation.")),
    ]

    @property
    def polynomial(self) -> np.polynomial.Polynomial:
        """
        Polynomial transformation.
        """
        return np.polynomial.Polynomial(self.polynomial_coefficients)


class MagnitudeTransformMatrix(BaseModel):
    """
    Class for magnitude transformation from one set of passbands to another that is best
    represented by a matrix.
    """

    name: str
    from_passbands: Annotated[
        list[str],
        Field(description="Passbands to transform from."),
    ]
    to_passbands: Annotated[
        list[str],
        Field(description="Passbands to transform to."),
    ]
    transformation_matrix: Annotated[
        list[list[float]],
        Field(
            description=(
                "Transformation matrix for the transformation. "
                "Shape should be (n_from, n_to)."
            )
        ),
    ]

    @property
    def array(self) -> np.ndarray:
        """
        Transformation matrix.
        """
        return np.array(self.transformation_matrix)


def _parse_transform_coefficients(
    input: Any,
) -> dict[tuple[str, str], MagnitudeTransformPolynomial]:
    """
    Parse the transform coefficients to ensure they are in the correct format.
    """
    if isinstance(input, str):
        return input

    if not isinstance(input, dict):
        raise ValueError(
            "Transform coefficients must be a dictionary of "
            "magnitude transformations."
        )
    return {
        tuple(k.split(",")) if isinstance(k, str) else k: v for k, v in input.items()
    }


class MagnitudeSystemTransform(BaseModel):
    """
    Class for magnitude system transformation from one single
    system to another.
    """

    name: str
    reference: Annotated[
        str,
        Field(
            description=(
                "Reference for the transformation, e.g., a " "paper or documentation."
            )
        ),
    ]
    from_system: MagnitudeSystem = Field(
        description="The magnitude system to transform from."
    )
    to_system: MagnitudeSystem = Field(
        description="The magnitude system to transform to."
    )
    transform_information: (
        Annotated[
            dict[tuple[str, str], MagnitudeTransformPolynomial],
            Field(
                description=(
                    "Coefficients for the transformation. The "
                    "number of coefficients depends on the transformation."
                )
            ),
            BeforeValidator(
                _parse_transform_coefficients,
            ),
        ]
        | Annotated[
            MagnitudeTransformMatrix,
            Field(
                description=(
                    "Transformation matrix for the transformation. "
                    "Shape should be (n_from, n_to)."
                )
            ),
        ]
    )


class PanStarrs1ToJohnsonCousinsMixin:
    """
    Implementation of the Pan-STARRS1 to Johnson-Cousins transformation.
    """

    def __call__(
        self, from_magnitudes: np.typing.ArrayLike, ps1_band_for_V: str = "gp1"
    ) -> np.ndarray:
        """
        Transform Pan-STARRS1 magnitudes to Johnson-Cousins magnitudes.

        Parameters
        ----------
        from_magnitudes : np.ArrayLike
            Pan-STARRS1 magnitudes to transform. The shape should either be (6,)
            or (n, 6), where n is the number of magnitudes in the Pan-STARRS1 system
            you wish to transform to Johnson-Cousins system.

        """
        # Convert to numpy array
        from_magnitudes = np.asarray(from_magnitudes)

        # The Pan-STARRS1 transformation is a polynomial of order 2
        # in which the independent variable, which they call "x",
        # is g_p1 - r_p1.
        from_band_index = {
            band: index for index, band in enumerate(self.from_system.passbands)
        }
        color = from_magnitudes[..., 0] - from_magnitudes[..., 1]
        if len(from_magnitudes.shape) == 1:
            from_magnitudes = from_magnitudes[np.newaxis, :]
        to_mags = np.zeros(
            from_magnitudes.shape[:-1] + (len(self.to_system.passbands),)
        )
        for index, jc_band in enumerate(self.to_system.passbands):
            to_band_name = jc_band
            match to_band_name:
                case "B":
                    from_band_name = "gp1"

                case "V":
                    # There are three options in the Pan-STARRS1 paper
                    # for the V band: g_p1, r_p1, and w_p1.
                    from_band_name = ps1_band_for_V
                case "Rc":
                    from_band_name = "rp1"
                case "Ic":
                    from_band_name = "ip1"
                case _:  # pragma: no cover
                    raise ValueError(
                        f"Unknown band name {to_band_name} for "
                        f"transformation from Pan-STARRS1 to Johnson-Cousins."
                    )

            to_mags[..., index] = (
                self.transform_information[(from_band_name, to_band_name)].polynomial(
                    color
                )
                + from_magnitudes[..., from_band_index[from_band_name]]
            )
        return np.squeeze(to_mags)


class MatrixTransformMixin:
    """
    Implementation of a matrix-based magnitude transformation.
    """

    def __call__(self, from_magnitudes: np.typing.ArrayLike) -> np.ndarray:
        """
        Transform magnitudes using a matrix transformation.

        Parameters
        ----------
        from_magnitudes : np.ArrayLike
            Magnitudes to transform. The shape should be (n, m), where n is the number
            of magnitudes and m is the number of passbands in the from_system. There
            must be an entry for each passband in the from_system, even if all of the
            entries for a particular passband are zero.
        """
        # Convert to numpy array
        from_magnitudes = np.asarray(from_magnitudes)

        # Ensure the input shape matches the expected number of passbands. The
        # +1 is for the constant term in the transforms
        if from_magnitudes.shape[0] != len(self.from_system.passbands) + 1:
            raise ValueError(
                f"Input shape {from_magnitudes.shape} does not match the number "
                f"of passbands in the from_system ({len(self.from_system.passbands)})."
            )

        # Perform the matrix multiplication
        return self.transform_information.array @ from_magnitudes


class PanStarrs1ToJohnsonCousins(
    PanStarrs1ToJohnsonCousinsMixin, MagnitudeSystemTransform
):
    """
    Class for transforming Pan-STARRS1 magnitudes to Johnson-Cousins magnitudes.
    """

    @classmethod
    def load(cls) -> "PanStarrs1ToJohnsonCousins":
        """
        Load the Pan-STARRS1 to Johnson-Cousins transformation from a file.
        """
        # Load the transformation from a file
        path = Path(get_pkg_data_path("data/PS1_to_JC.json"))
        return cls.model_validate_json(path.read_text())


class USNOPrimeToSDSSDR7(MatrixTransformMixin, MagnitudeSystemTransform):
    """
    Class for transforming USNO Prime magnitudes to SDSS DR7 magnitudes.
    """

    @classmethod
    def load(cls) -> "USNOPrimeToSDSSDR7":
        """
        Load the USNO Prime to SDSS DR7 transformation from a file.
        """
        # Load the transformation from a file
        path = Path(get_pkg_data_path("data/USNO_SDSS_to_SDSS_DR7.json"))
        return cls.model_validate_json(path.read_text())


def transform_apass_bands(table, apply_sdssdr7_transform: bool = False):
    """
    A function for transforming from the native APASS bassbands to Johnson/Cousins
    R and I bands.

    Parameters
    ----------
    table : `astropy.table.Table`
        Table of catalog information. This is assumed to be in "passband" format
    """
    # Putting this here to avoid a circular import
    from .magnitude_transforms import filter_transform

    # Set up for input to Jester transform
    use_columns = dict(
        g="mag_SG",
        r="mag_SR",
        i="mag_SI",
    )

    # The Jester transformation really for transformation from the ugriz on the
    # 2.5m SDSS telescope to the Johnson-Cousins system.
    #
    # APASS is using u'g'r'i'z' (well, only g'r'i' in APASS) and so I *think*
    # that the right thing to do is to transform the APASS g'r'i' to the SDSS DR7
    # system, and then apply jester.
    if apply_sdssdr7_transform:
        # Load the transformation
        transform = USNOPrimeToSDSSDR7.load()

        # Get the g', r', i' columns
        gp = table["mag_SG"]
        rp = table["mag_SR"]
        ip = table["mag_SI"]
        up = np.zeros_like(gp)  # Placeholder for u' band, not used in APASS
        zp = np.zeros_like(gp)  # Placeholder for z' band, not used in APASS
        ones = np.ones_like(gp)  # Placeholder for the constant term
        # Prepare the data for the transformation
        input_mags = np.asarray([up, gp.data, rp.data, ip.data, zp, ones])
        # Transform the APASS g'r'i' to SDSS DR7
        sdss_mags = transform(input_mags)
        new_columns = {
            "mag_SG_tmp": sdss_mags[1, :],
            "mag_SR_tmp": sdss_mags[2, :],
            "mag_SI_tmp": sdss_mags[3, :],
        }
        for k, v in new_columns.items():
            table[k] = v

        use_columns = dict(
            g="mag_SG_tmp",
            r="mag_SR_tmp",
            i="mag_SI_tmp",
        )

    table["mag_RC"] = filter_transform(
        table,
        output_filter="R",
        **use_columns,
        transform="jester",
    )
    table["mag_IC"] = filter_transform(
        table,
        output_filter="I",
        **use_columns,
        transform="jester",
    )

    # Yes, this is dumb. You fix it if you want it less dumb.
    table["mag_R"] = table["mag_RC"]
    table["mag_I"] = table["mag_IC"]
    # The dumbness has ended for now

    # Remove the temporary columns if they were created
    if apply_sdssdr7_transform:
        del table["mag_SG_tmp", "mag_SR_tmp", "mag_SI_tmp"]

    return table


def transform_refcat2_bands(table):
    """
    A function for transforming from the native RefCat2 bassbands to Johnson/Cousins
    BVRI bands.

    Parameters
    ----------
    table : `astropy.table.Table`
        Table of catalog information. This is assumed to be in "passband" format
    """
    transform = PanStarrs1ToJohnsonCousins.load()

    # Prepare the data for the transformation
    num_rows = len(table)

    transform_input = np.array(
        [
            table["mag_SG"],
            table["mag_SR"],
            table["mag_SI"],
            table["mag_SZ"],
            [0] * num_rows,  # Placeholder for the y_p1 band
            [0] * num_rows,  # Placeholder for the w_p1 band
        ]
    )

    transformed_data = transform(
        transform_input.T,
        ps1_band_for_V="gp1",
    )

    table["mag_B"] = transformed_data[:, 0]
    table["mag_V"] = transformed_data[:, 1]
    table["mag_RC"] = transformed_data[:, 2]
    table["mag_IC"] = transformed_data[:, 3]

    # Yes, this is dumb. You fix it if you want it less dumb.
    # We should only have one name for RC and IC, not two...
    table["mag_R"] = table["mag_RC"]
    table["mag_I"] = table["mag_IC"]
    # The dumbness has ended for now

    return table
