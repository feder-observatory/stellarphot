from enum import StrEnum
from pathlib import Path
from typing import Annotated

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
            lambda x: {
                tuple(k.split(",")) if isinstance(k, str) else k: v
                for k, v in x.items()
            }
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
            or (n, 6), where n is the number of magnitudesin the Pan-STARRS1 system
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
