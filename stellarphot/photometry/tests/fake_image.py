import astropy.io.fits as fits
import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.nddata import CCDData
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table, vstack
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from photutils.datasets import make_model_image, make_noise_image


class FakeImage:
    """
    Creates a fake image with a set of sources using the datafile stored
    at `data/test_sources.csv`.

    Parameters
    ----------

    noise_dev : float, optional
        The standard deviation of the noise in the image.

    seed : int, optional
        The seed to use for the random number generator. If not specified,
        the seed will be randomly generated.

    fwhm : float, optional
        The full width at half maximum (FWHM) of the sources in pixels. If
        not specified, the FWHM will be set to value in the data file.

    n_repeats_per_side : int, optional
        The number of times to repeat the sources in each direction.
    """

    def __init__(self, noise_dev=1.0, seed=None, fwhm=None, n_repeats_per_side=1):
        self.image_shape = np.array([400, 500])
        data_file = get_pkg_data_filename("data/test_sources.csv")
        self._sources = Table.read(data_file)
        self.input_fwhm = fwhm
        if fwhm is not None:
            # Set the FWHM of the sources to the specified value
            self._sources["x_stddev"] = gaussian_fwhm_to_sigma * fwhm
            self._sources["y_stddev"] = gaussian_fwhm_to_sigma * fwhm
        if n_repeats_per_side > 1:
            amplitude_scales = np.logspace(
                start=np.log10(0.1), stop=np.log10(60.0), num=n_repeats_per_side**2
            )
            # Reshape to n_repeats x n_repeats
            amplitude_scales = amplitude_scales.reshape(
                (n_repeats_per_side, n_repeats_per_side)
            )
            extra_sources = []
            for i in range(n_repeats_per_side):
                for j in range(n_repeats_per_side):
                    if i == 0 and j == 0:
                        continue
                    new_sources = self._sources.copy()
                    new_sources["x_mean"] = (
                        self._sources["x_mean"] + i * self.image_shape[1]
                    )
                    new_sources["y_mean"] = (
                        self._sources["y_mean"] + j * self.image_shape[0]
                    )
                    # Tinker with the amplitudes a bit to get more variation
                    # in the image
                    new_sources["amplitude"] = (
                        self._sources["amplitude"] * amplitude_scales[i, j]
                    )
                    # Add the new sources to the list
                    extra_sources.append(new_sources)
            # Concatenate the new sources to the original sources
            self._sources = vstack([self._sources] + extra_sources)
            # Reset the image shape -- this needs to be after the loop above so that
            # the new source positions are based on the original image shape
            self.image_shape = self.image_shape * n_repeats_per_side

        self.mean_noise = self.sources["amplitude"].max() / 100
        self.noise_dev = noise_dev
        self._stars = make_gaussian_sources_image(tuple(self.image_shape), self.sources)
        self._noise = make_noise_image(
            self._stars.shape, mean=self.mean_noise, stddev=noise_dev, seed=seed
        )
        # Sky background per pixel should be the mean level of the noise.
        self._sources["sky_per_pix_avg"] = self.mean_noise

    @property
    def sources(self):
        """
        Return the table of sources used to create the fake image.
        """
        return self._sources

    @property
    def image(self):
        """
        Return the fake image.
        """
        return self._stars + self._noise


class FakeCCDImage(CCDData):
    # Generates a fake CCDData object for testing purposes.
    def __init__(self, *args, **kwargs):
        # Pull off the seed argument if it exists.
        seed = kwargs.pop("seed", None)
        # Pull off the fwhm argument if it exists.
        fwhm = kwargs.pop("fwhm", None)
        # Pull off the noise_dev argument if it exists.
        noise_dev = kwargs.pop("noise_dev", 1.0)
        # Pull off the n_repeats argument if it exists.
        n_repeats = kwargs.pop("n_repeats", 1)

        # If no arguments are passed, use the default FakeImage.
        # This dodge is necessary because otherwise we can't copy the CCDData
        # object apparently.
        if (len(args) == 0) and (len(kwargs) == 0):
            base_data = FakeImage(
                seed=seed, fwhm=fwhm, noise_dev=noise_dev, n_repeats_per_side=n_repeats
            )
            super().__init__(base_data.image.copy(), unit="adu")

            # Append attributes from the base data object.
            self.sources = base_data.sources.copy()
            self.noise_dev = base_data.noise_dev
            self.mean_noise = base_data.mean_noise
            self.image_shape = base_data.image_shape.copy()

            # Add some additional features to the CCDData object, like
            # a header and the sources used to create the image.abs
            self.header = fits.Header()
            self.header["OBJECT"] = "Test Object"
            self.header["EXPOSURE"] = 1.0
            self.header["DATE-OBS"] = "2018-01-01T00:00:00.0"
            self.header["AIRMASS"] = 1.2
            self.header["FILTER"] = "V"

            # Set up a  WCS header for the CCDData object.
            (size_y, size_x) = base_data.image_shape
            pixel_scale = 0.75  # arcseconds per pixel
            ra_center = 283.6165
            dec_center = 33.05857
            w = WCS(naxis=2)
            w.wcs.crpix = [
                size_x / 2,
                size_y / 2,
            ]  # Reference pixel (center of the image)
            w.wcs.cdelt = [
                -pixel_scale / 3600,
                pixel_scale / 3600,
            ]  # Pixel scale in degrees per pixel
            w.wcs.crval = [
                ra_center,
                dec_center,
            ]  # RA and Dec of the reference pixel in degrees
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Coordinate type (TAN projection)
            # Rotate image to be slightly off of horizontal
            north_angle_deg = 8.4
            w.wcs.pc = [
                [
                    np.cos(np.radians(north_angle_deg)),
                    -np.sin(np.radians(north_angle_deg)),
                ],
                [
                    np.sin(np.radians(north_angle_deg)),
                    np.cos(np.radians(north_angle_deg)),
                ],
            ]

            self.wcs = w
            self.header.update(w.to_header())

    def drop_wcs(self):
        # Convenience function to remove WCS information from the CCDData object
        # for testing purposes.
        self.wcs = None
        wcs_keywords = [
            "CTYPE",
            "CRPIX",
            "CRVAL",
            "CDELT",
            "CUNIT",
            "CD1_",
            "CD2_",
            "PC1_",
            "PC2_",
        ]
        for keyword in wcs_keywords:
            matching_keys = [key for key in self.header.keys() if keyword in key]
            for key in matching_keys:
                del self.header[key]


def shift_FakeCCDImage(ccd_data, x_shift, y_shift):
    """
    Create a test CCD image based on the first test image with the central
    positions shifted by the given amount.  To prevent any sources being
    shifted off the edge of the image, an error is thrown if this is
    happening.

    WCS information in the CCDData is also updated to reflect the shift.

    Parameters:
        ccd_data : `FakeCCDImage`
            The original CCDData object.

        x_shift : int
            The amount of shift in x direction (in pixels)

        y_shift : int
            The amount of shift in y direction (in pixels)

    Returns:
        `FakeCCDImage`: A new CCDData object.
    """
    # Copy WCS from original CCDData object
    shifted_ccd_data = ccd_data.copy()
    for key, val in ccd_data.__dict__.items():
        try:
            shifted_ccd_data.__dict__[key] = val.copy()
        except AttributeError:
            shifted_ccd_data.__dict__[key] = val
    shifted_wcs = ccd_data.wcs.deepcopy()

    # Calculate the new RA and Dec center after shifting
    x_shift = int(x_shift)
    y_shift = int(y_shift)
    ra_center_shifted, dec_center_shifted = shifted_wcs.all_pix2world(
        (shifted_ccd_data.data.shape[1]) / 2 + x_shift,
        (shifted_ccd_data.data.shape[0]) / 2 + y_shift,
        0,
    )

    # Shift source positions
    shifted_ccd_data.sources["x_mean"] -= x_shift
    shifted_ccd_data.sources["y_mean"] -= y_shift

    # Check shifted sources are still on the image
    if (
        (np.any(shifted_ccd_data.sources["x_mean"] < 0))
        | (np.any(shifted_ccd_data.sources["x_mean"] > shifted_ccd_data.data.shape[1]))
        | (np.any(shifted_ccd_data.sources["y_mean"] < 0))
        | (np.any(shifted_ccd_data.sources["y_mean"] > shifted_ccd_data.data.shape[0]))
    ):
        raise ValueError("Sources shifted off the edge of the image.")

    # Update the new RA and Dec center in the shifted WCS
    shifted_wcs.wcs.crval = [ra_center_shifted, dec_center_shifted]
    shifted_ccd_data.wcs = shifted_wcs

    # Make image
    srcs = make_gaussian_sources_image(
        tuple(shifted_ccd_data.image_shape), shifted_ccd_data.sources
    )
    background = make_noise_image(
        srcs.shape, mean=shifted_ccd_data.mean_noise, stddev=shifted_ccd_data.noise_dev
    )
    shifted_ccd_data.data = srcs + background

    return shifted_ccd_data


def make_gaussian_sources_image(shape, source_table, oversample=1.0):
    """
    Make an image containing 2D Gaussian sources.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the output 2D image.

    source_table : `~astropy.table.Table`
        Table of parameters for the Gaussian sources. Each row of the
        table corresponds to a Gaussian source whose parameters are
        defined by the column names. With the exception of ``'flux'``,
        column names that do not match model parameters will be ignored
        (flux will be converted to amplitude). If both ``'flux'`` and
        ``'amplitude'`` are present, then ``'flux'`` will be ignored.
        Model parameters not defined in the table will be set to the
        default value.

    oversample : float, optional
        The sampling factor used to discretize the models on a pixel
        grid. If the value is 1.0 (the default), then the models will
        be discretized by taking the value at the center of the pixel
        bin. Note that this method will not preserve the total flux of
        very small sources. Otherwise, the models will be discretized by
        taking the average over an oversampled grid. The pixels will be
        oversampled by the ``oversample`` factor.

    Note
    ----

    The body of this function, including the docstring, are copy/pasted from
    photutils 1.13.0 because the function make_gaussian_sources_image was
    deprecated in that version. Link:

    https://github.com/astropy/photutils/blob/main/photutils/datasets/images.py#L388

    Returns
    -------
    image : 2D `~numpy.ndarray`
        Image containing 2D Gaussian sources.
    """
    model = Gaussian2D(x_stddev=1, y_stddev=1)

    if "x_stddev" in source_table.colnames:
        xstd = source_table["x_stddev"]
    else:
        xstd = model.x_stddev.value  # default
    if "y_stddev" in source_table.colnames:
        ystd = source_table["y_stddev"]
    else:
        ystd = model.y_stddev.value  # default

    colnames = source_table.colnames
    if "flux" in colnames and "amplitude" not in colnames:
        source_table = source_table.copy()
        source_table["amplitude"] = source_table["flux"] / (2.0 * np.pi * xstd * ystd)

    return make_model_image(
        shape,
        model,
        source_table,
        x_name="x_mean",
        y_name="y_mean",
        discretize_oversample=oversample,
    )
