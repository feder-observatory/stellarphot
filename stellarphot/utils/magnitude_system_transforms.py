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
        self,
        from_magnitudes: np.typing.ArrayLike,
        ps1_band_for_V: str = "gp1",
        from_magnitude_errors: np.typing.ArrayLike | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Transform Pan-STARRS1 magnitudes to Johnson-Cousins magnitudes.

        Parameters
        ----------
        from_magnitudes : np.ArrayLike
            Pan-STARRS1 magnitudes to transform. The shape should either be (6,)
            or (n, 6), where n is the number of magnitudes in the Pan-STARRS1 system
            you wish to transform to Johnson-Cousins system.

        ps1_band_for_V : str, optional
            Pan-STARRS1 band to use as the base magnitude for the V band.

        from_magnitude_errors : np.ArrayLike, optional
            Errors in the Pan-STARRS1 magnitudes, same shape as
            ``from_magnitudes``. When given, the return value is the tuple
            ``(magnitudes, errors)``, where the errors combine the input
            errors propagated through each transform with the published
            residual of that transform, in quadrature.
        """
        # Convert to numpy array
        from_magnitudes = np.asarray(from_magnitudes)

        # The Pan-STARRS1 transformation is a polynomial of order 2
        # in which the independent variable, which they call "x",
        # is g_p1 - r_p1.
        from_band_index = {
            band: index for index, band in enumerate(self.from_system.passbands)
        }
        missing_bands = [band for band in ("gp1", "rp1") if band not in from_band_index]
        if missing_bands:
            raise ValueError(
                "Cannot compute the Pan-STARRS1 g-r color needed for this "
                "transformation because the following required passband(s) "
                f"are missing from from_system.passbands: {missing_bands}."
            )
        color = (
            from_magnitudes[..., from_band_index["gp1"]]
            - from_magnitudes[..., from_band_index["rp1"]]
        )
        if len(from_magnitudes.shape) == 1:
            from_magnitudes = from_magnitudes[np.newaxis, :]
        if from_magnitude_errors is not None:
            from_magnitude_errors = np.asarray(from_magnitude_errors).reshape(
                from_magnitudes.shape
            )
        to_mags = np.zeros(
            from_magnitudes.shape[:-1] + (len(self.to_system.passbands),)
        )
        to_errors = np.zeros_like(to_mags)
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

            transform = self.transform_information[(from_band_name, to_band_name)]
            to_mags[..., index] = (
                transform.polynomial(color)
                + from_magnitudes[..., from_band_index[from_band_name]]
            )

            if from_magnitude_errors is not None:
                # The transform is base magnitude plus polynomial in the
                # g - r color, so the partial with respect to each input
                # band is the polynomial's derivative through the color
                # (+1 for gp1, -1 for rp1) plus one for the base band.
                dpoly_dcolor = transform.polynomial.deriv()(color)
                partials = np.zeros_like(from_magnitudes)
                partials[..., from_band_index["gp1"]] += dpoly_dcolor
                partials[..., from_band_index["rp1"]] -= dpoly_dcolor
                partials[..., from_band_index[from_band_name]] += 1.0
                to_errors[..., index] = np.sqrt(
                    np.sum((partials * from_magnitude_errors) ** 2, axis=-1)
                    + transform.residual**2
                )

        if from_magnitude_errors is not None:
            return np.squeeze(to_mags), np.squeeze(to_errors)
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
            Magnitudes to transform. The shape should be (p + 1,) or (p + 1, n),
            where p is the number of passbands in the from_system and n is the
            number of stars. The rows are the magnitudes in the order of the
            from_system passbands, followed by a final row of ones for the
            constant term in the transform. There must be a row for each
            passband in the from_system, even if all of the entries for a
            particular passband are zero.
        """
        # Convert to numpy array
        from_magnitudes = np.asarray(from_magnitudes)

        # Ensure the input shape matches the expected number of passbands. The
        # +1 is for the constant term in the transforms
        if from_magnitudes.shape[0] != len(self.from_system.passbands) + 1:
            n_bands = len(self.from_system.passbands)
            raise ValueError(
                f"Input shape {from_magnitudes.shape} is invalid: expected "
                f"{n_bands} passband rows plus a final row of ones "
                f"({n_bands + 1} rows total)."
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
    from .magnitude_transforms import _JESTER_RMS, _JESTER_TRANSFORMS, filter_transform

    # Set up for input to Jester transform
    use_columns = dict(
        g="mag_SG",
        r="mag_SR",
        i="mag_SI",
    )

    # Errors are propagated only when the catalog supplies them for every
    # native band the transform uses.
    error_columns = dict(
        g="mag_error_SG",
        r="mag_error_SR",
        i="mag_error_SI",
    )
    have_errors = all(column in table.colnames for column in error_columns.values())
    if have_errors:
        error_values = {band: table[column] for band, column in error_columns.items()}
        # Effective linear coefficients of the native g, r and i in each
        # output band; without the SDSS DR7 step these are just the Jester
        # coefficients.
        effective_coeffs = {
            band: np.asarray(_JESTER_TRANSFORMS[band][:3], dtype=float)
            for band in ("R", "I")
        }

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

        if have_errors:
            # The matrix step makes the SDSS g, r and i magnitudes
            # correlated -- each is a mix of the same native bands -- so the
            # errors cannot be propagated through it and Jester separately.
            # Compose the two linear maps instead: the effective coefficient
            # of each native band is the Jester row through the matrix.
            matrix = transform.transform_information.array
            for band in ("R", "I"):
                jester_row = np.zeros(matrix.shape[0])
                # Rows 1, 2 and 3 of the matrix output are g, r and i, the
                # same indexing used for sdss_mags above.
                jester_row[1:4] = _JESTER_TRANSFORMS[band][:3]
                effective_coeffs[band] = (jester_row @ matrix)[1:4]

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

    if have_errors:
        # The transforms are linear in the native g, r and i, so the errors
        # propagate through the squared coefficients, and the rms residual
        # of the Jester transform adds in quadrature.
        for band, native in (("R", "RC"), ("I", "IC")):
            coeff_g, coeff_r, coeff_i = effective_coeffs[band]
            table[f"mag_error_{native}"] = np.sqrt(
                (coeff_g * error_values["g"]) ** 2
                + (coeff_r * error_values["r"]) ** 2
                + (coeff_i * error_values["i"]) ** 2
                + _JESTER_RMS[band] ** 2
            )

    # Yes, this is dumb. You fix it if you want it less dumb.
    table["mag_R"] = table["mag_RC"]
    table["mag_I"] = table["mag_IC"]
    if have_errors:
        table["mag_error_R"] = table["mag_error_RC"]
        table["mag_error_I"] = table["mag_error_IC"]
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

    error_columns = ["mag_error_SG", "mag_error_SR", "mag_error_SI", "mag_error_SZ"]
    have_errors = all(column in table.colnames for column in error_columns)
    if have_errors:
        zeros = np.zeros(num_rows)
        error_input = np.array(
            [
                table["mag_error_SG"],
                table["mag_error_SR"],
                table["mag_error_SI"],
                table["mag_error_SZ"],
                zeros,  # Placeholder for the y_p1 band
                zeros,  # Placeholder for the w_p1 band
            ]
        )
        transformed_data, transformed_errors = transform(
            transform_input.T,
            ps1_band_for_V="gp1",
            from_magnitude_errors=error_input.T,
        )
        transformed_errors = np.atleast_2d(transformed_errors)
    else:
        transformed_data = transform(
            transform_input.T,
            ps1_band_for_V="gp1",
        )

    # A one-row table comes back squeezed to a 1-d array
    transformed_data = np.atleast_2d(transformed_data)

    table["mag_B"] = transformed_data[:, 0]
    table["mag_V"] = transformed_data[:, 1]
    table["mag_RC"] = transformed_data[:, 2]
    table["mag_IC"] = transformed_data[:, 3]

    if have_errors:
        table["mag_error_B"] = transformed_errors[:, 0]
        table["mag_error_V"] = transformed_errors[:, 1]
        table["mag_error_RC"] = transformed_errors[:, 2]
        table["mag_error_IC"] = transformed_errors[:, 3]

    # Yes, this is dumb. You fix it if you want it less dumb.
    # We should only have one name for RC and IC, not two...
    table["mag_R"] = table["mag_RC"]
    table["mag_I"] = table["mag_IC"]
    if have_errors:
        table["mag_error_R"] = table["mag_error_RC"]
        table["mag_error_I"] = table["mag_error_IC"]
    # The dumbness has ended for now

    return table
