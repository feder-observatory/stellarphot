from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from astropy.utils.data import get_pkg_data_path
from pydantic import BaseModel, BeforeValidator, Field

__all__ = [
    "MagnitudeSystemNames",
    "MagnitudeSystem",
    "MagnitudeTransform",
    "MagnitudeSystemTransform",
    "PanStarrs1ToJohnsonCousins",
]


class MagnitudeSystemNames(StrEnum):
    """Enum for different magnitude systems."""

    AB = "AB"
    Vega = "Vega"
    JC = "JC"
    SDSSDR7 = "SDSSDR7"
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


class MagnitudeTransform(BaseModel):
    """
    Class for magnitude transformation from one set of passbands to another.
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


def _parse_transform_coefficients(
    input: Any,
) -> dict[tuple[str, str], MagnitudeTransform]:
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
    transform_coefficients: Annotated[
        dict[tuple[str, str], MagnitudeTransform],
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
                self.transform_coefficients[(from_band_name, to_band_name)].polynomial(
                    color
                )
                + from_magnitudes[..., from_band_index[from_band_name]]
            )
        return np.squeeze(to_mags)


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


def transform_apass_bands(table):
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

    table["mag_RC"] = filter_transform(
        table,
        output_filter="R",
        g="mag_SG",
        r="mag_SR",
        i="mag_SI",
        transform="jester",
    )
    table["mag_IC"] = filter_transform(
        table,
        output_filter="I",
        g="mag_SG",
        r="mag_SR",
        i="mag_SI",
        transform="jester",
    )

    # Yes, this is dumb. You fix it if you want it less dumb.
    table["mag_R"] = table["mag_RC"]
    table["mag_I"] = table["mag_IC"]
    # The dumbness has ended for now

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
