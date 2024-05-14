import warnings

import numpy as np
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.stats import sigma_clipped_stats
from astropy.utils import lazyproperty
from photutils.centroids import centroid_2dg, centroid_com
from photutils.profiles import CurveOfGrowth, RadialProfile

from stellarphot.photometry import calculate_noise

__all__ = ["find_center", "CenterAndProfile"]


def find_center(image, center_guess, cutout_size=30, max_iters=10, match_limit=3):
    """
    Find the centroid of a star from an initial guess of its position. Originally
    written to find star from a mouse click.

    Parameters
    ----------

    image : `astropy.nddata.CCDData` or numpy array
        Image containing the star.

    center_guess : array or tuple
        The position, in pixels, of the initial guess for the position of
        the star. The coordinates should be horizontal first, then vertical,
        i.e. opposite the usual Python convention for a numpy array.

    cutout_size : int, optional
        The default width of the cutout to use for finding the star.

    max_iters : int, optional
        Maximum number of iterations to go through in finding the center.

    match_limit : int, optional
        Maximum number of pixels to allow the COM centroid and Gaussian center to
        differ.

    Returns
    -------

    cen : array
        The position of the star, in pixels, as found by the centroiding
        algorithm.


    Raises
    ------
    RuntimeError
        If the centroiding algorithm fails to converge on a star eitehr because a cutout
        has only ``NaN`` values or because the centroiding algorithm fails to converge
        or because the centroid determined by ``centroid_com`` and ``centroid_2dg``
        differ by more than ``match_limit`` pixels.

    Notes
    -----
    This function tries to identify the centroid of a star in a small region around an
    image position. The original mtivation was to find the star near a mouse click on
    an image. The approach is to generate a cutout around the initial guess position,
    then use the centroid_com function from photutils to find the centroid of the star.
    A new cutout is then generated around the new centroid position, and the process
    is repeated until the centroid converges.

    Convergence is determined by three criteria:

    1. The centroid of the cutout must be within 3 pixels of the center of the cutout.
    2. The centroid of the cutout must be within 0.1 pixels of the previous centroid.
    3. The first two criteria must be met within the maximum number of iterations.

    If the first two criteria are satisfied then the centroid is found by fitting a
    Gaussian to the cutout. If the two centroids differ by more than match_limit pixels
    then an error is raised.
    """
    pad = cutout_size // 2
    x, y = center_guess

    # Keep track of iterations
    cnt = 0

    # Grab the cutout...
    sub_data = Cutout2D(image, center_guess, (cutout_size, cutout_size), mode="trim")
    # ...do stats on it...
    _, sub_med, _ = sigma_clipped_stats(sub_data.data)
    # ...and centroid.

    # Exclude negative pixels from initial centroid. If there is a dim star this helps
    # ensure the star ends up centered since pixels with negative values are likely
    # background.
    # See also Howell, Handbook of CCD Astronomy, 2nd ed., p. 105
    mask = (sub_data.data - sub_med) < 0
    x_cm, y_cm = centroid_com(sub_data.data - sub_med, mask=mask)

    # Translate centroid back to original image
    cen = np.array(sub_data.to_original_position((x_cm, y_cm)))

    # ceno is the "original" center guess, set it to something nonsensical here
    ceno = np.array([-100, -100])

    while cnt <= max_iters and (  # Iteration limit has not been reached
        np.abs(np.array([x_cm, y_cm]) - pad).max() > 3  # Centroid > 3 pix from center
        or np.abs(cen - ceno).max() > 0.1  # Centroid has not converged
    ):
        try:
            sub_data = Cutout2D(image, cen, (cutout_size, cutout_size), mode="trim")
        except NoOverlapError as err:
            raise RuntimeError(
                f"Centroid finding failed, previous was {ceno}, current is {cen}"
            ) from err
        _, sub_med, _ = sigma_clipped_stats(sub_data.data)

        mask = (sub_data.data - sub_med) < 0
        x_cm, y_cm = centroid_com(sub_data.data - sub_med, mask=mask)
        ceno = cen
        cen = np.array(sub_data.to_original_position((x_cm, y_cm)))
        if not np.all(~np.isnan(cen)):
            raise RuntimeError(
                f"Centroid finding failed, previous was {ceno}, current is {cen}"
            )
        cnt += 1

    # Is we hit the max number of iterations, raise an error
    if max_iters > 1 and cnt > max_iters:
        raise RuntimeError("Centroid finding did not converge")

    # Get the final centroid position by fitting a gaussian
    # We may not have converged on a star, so capture any warning about a
    # bad fit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ceng = centroid_2dg(sub_data.data - sub_med)
    ceng = np.array(sub_data.to_original_position(ceng))

    # Confirm that we actually found a star by comparing two centroiding
    # methods
    if np.linalg.norm(cen - ceng) > match_limit:
        raise RuntimeError(
            "Centroid did not converge on a star. "
            f"Got {cen} from centroid_com and {ceng} from centroid_2dg."
        )

    return ceng


class CenterAndProfile:
    """
    Class to determine center of and hold radial profile information for a star.

    Parameters
    ----------
    data : `astropy.nddata.CCDData` or numpy array
        Image data

    center_approx : list-like
        x, y position of the center in pixel coordinates, i.e. horizontal
        coordinate then vertical.

    cutout_size : int, optional
        Width of the rectangular image cutout to use in looking for a star.

    profile_radius : int, optional
        Maximum radius to use in constructing the profile.

    max_iters : int, optional
        Maximum number of iterations to go through in finding the center.
    """

    def __init__(
        self, data, center_approx, cutout_size=30, profile_radius=None, max_iters=10
    ):
        self._cen = find_center(data, center_approx, cutout_size=cutout_size)
        self._data = data
        self._cutout = Cutout2D(data, self._cen, (cutout_size, cutout_size))
        self._max_iters = max_iters
        if profile_radius is None:
            profile_radius = cutout_size // 2

        radii = np.linspace(0, profile_radius, profile_radius + 1)
        # Get a rough profile with rough background subtraction -- note that
        # NO background subtraction does not work.
        background = sigma_clipped_stats(self._cutout.data)[1]
        self._radial_profile = RadialProfile(self._data - background, self._cen, radii)

        self._sky_area = None

        # Do proper background subtraction for the final radial profile
        self._radial_profile = RadialProfile(
            data - self.sky_pixel_value, self._cen, radii
        )

    @property
    def center(self):
        """
        x, y position of the center of the star.
        """
        return self._cen

    @property
    def FWHM(self):
        """
        Full-width half-max of the radial profile.
        """
        return self.radial_profile.gaussian_fwhm

    @property
    def HWHM(self):
        """
        Half-width half-max of the radial profile.
        """
        return self.FWHM / 2

    @property
    def cutout(self):
        """
        Cutout image around the star.
        """
        return self._cutout

    @property
    def radial_profile(self):
        """
        Radial profile of the star.
        """
        return self._radial_profile

    @property
    def max_iters(self):
        """
        Maximum number of iterations to go through in finding the center.
        """
        return self._max_iters

    @lazyproperty
    def normalized_profile(self):
        """
        Radial profile scaled to have a maximum of 1.
        """
        # This seems to be what photutils does under the hood
        return self.radial_profile.profile / self.radial_profile.profile.max()

    @lazyproperty
    def pixel_values_in_profile(self):
        """
        Pixel values in the radial profile.
        """
        radii = []
        pixel_values = []
        for rad, ap in zip(
            self.radial_profile.radius, self.radial_profile.apertures, strict=True
        ):
            ap_mask = ap.to_mask(method="center")
            ap_data = ap_mask.multiply(self._data)
            good_data = ap_data != 0
            pixel_values.extend(ap_data[good_data].flatten())
            radii.extend([rad] * good_data.sum())
        radii = np.array(radii)
        pixel_values = np.array(pixel_values)
        return radii, pixel_values

    @lazyproperty
    def curve_of_growth(self):
        """
        Curve of growth for the star.
        """
        self._cog = CurveOfGrowth(
            self._data - self.sky_pixel_value,
            self.center,
            self.radial_profile.radii + 1,
        )
        return self._cog

    @lazyproperty
    def sky_pixel_value(self):
        """
        Pixel values for the sky, i.e. more than 3 FWHM from the star.
        """
        grid_x, grid_y = np.mgrid[: self.cutout.shape[0], : self.cutout.shape[1]]
        x_s, y_s = self.cutout.to_cutout_position(self.center)
        dist_from_star = np.sqrt((grid_x - x_s) ** 2 + (grid_y - y_s) ** 2)
        mask = dist_from_star > self.FWHM * 3
        self._sky_area = mask.sum()
        _, median, _ = sigma_clipped_stats(self.cutout.data[mask])
        return median

    @lazyproperty
    def sky_area(self):
        """
        Area of the sky annulus.
        """
        if self._sky_area is None:
            # sky area is set as a side effect of this....
            _ = self.sky_pixel_value

        return self._sky_area

    def noise(self, camera, exposure):
        """
        Noise in the star.

        Parameters
        ----------

        camera : `stellarphot.settings.Camera`
            Camera settings.

        exposure : float
            Exposure time in seconds.

        Returns
        -------
        noise : `numpy.ndarray` of float
            Noise calculated using the CCD equation.
        """
        return calculate_noise(
            camera,
            counts=self.curve_of_growth.profile,
            sky_per_pix=self.sky_pixel_value,
            aperture_area=self.curve_of_growth.area,
            annulus_area=0,
            exposure=exposure,
        )

    def snr(self, camera, exposure):
        """
        Signal to noise ratio of the star.

        Parameters
        ----------
        camera : `stellarphot.settings.Camera`
            Camera settings.

        exposure : float
            Exposure time in seconds.

        Returns
        -------
        snr : `numpy.ndarray` of float
            Signal to noise ratio.
        """
        return camera.gain * self.curve_of_growth.profile / self.noise(camera, exposure)
