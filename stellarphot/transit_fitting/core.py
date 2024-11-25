import warnings

import numpy as np
from astropy import units as u
from astropy.modeling.fitting import LevMarLSQFitter, _validate_model
from astropy.modeling.models import custom_model
from pydantic import BaseModel

# Functions below changed from private to public in astropy 5
try:
    from astropy.modeling.fitting import (
        fitter_to_model_params,
        model_to_fit_params,
    )
except ImportError:
    from astropy.modeling.fitting import (
        _fitter_to_model_params as fitter_to_model_params,
    )
    from astropy.modeling.fitting import (
        _model_to_fit_params as model_to_fit_params,
    )

from astropy.utils.exceptions import AstropyUserWarning

try:
    import batman
except ImportError:
    ImportError(
        "You must install the batman exoplanet package. Try:\n"
        "conda install batman-package\n"
        "or\n"
        "pip install batman-package"
    )

__all__ = ["VariableArgsFitter", "TransitModelOptions", "TransitModelFit"]


class VariableArgsFitter(LevMarLSQFitter):
    """
    A callable class that can be used to fit functions with arbitrary number of
    positional parameters.  This is a modified version of the
    astropy.modeling.fitting.LevMarLSQFitter fitter.

    """

    def __init__(self):
        super().__init__()

    # This is a straight copy-paste from the LevMarLSQFitter __call__.
    # The only modification is to allow any number of arguments.
    def __call__(
        self,
        model,
        *args,
        weights=None,
        maxiter=100,
        acc=1e-7,
        epsilon=1.4901161193847656e-08,
        estimate_jacobian=False,
    ):
        from scipy import optimize

        model_copy = _validate_model(model, self.supported_constraints)
        farg = (
            model_copy,
            weights,
        ) + args
        if model_copy.fit_deriv is None or estimate_jacobian:
            dfunc = None
        else:
            dfunc = self._wrap_deriv
        init_values0 = model_to_fit_params(model_copy)
        # This returns a tuple of (model_params, fitparam_indices, model_bounds),
        # where model_params is a numpy array, fitparam_indices is a list, and
        # model_bounds is a tuple of tuples.  The problem is that this doesn't
        # convert simply into an array within scipy.optimize.leastsq, when called.
        # So we handle model_bounds here first to the scipy.optimize.leastsq format.
        # can handle the list of initial values we pass in.
        init_values = np.concatenate(
            (
                np.asarray(init_values0[0]).flatten(),
                np.asarray(init_values0[1]).flatten(),
                np.asarray(init_values0[2]).flatten(),
            ),
            axis=None,
        )

        fitparams, cov_x, dinfo, mess, ierr = optimize.leastsq(
            self.objective_function,
            init_values,
            args=farg,
            Dfun=dfunc,
            col_deriv=model_copy.col_fit_deriv,
            maxfev=maxiter,
            epsfcn=epsilon,
            xtol=acc,
            full_output=True,
        )
        fitter_to_model_params(model_copy, fitparams)
        self.fit_info.update(dinfo)
        self.fit_info["cov_x"] = cov_x
        self.fit_info["message"] = mess
        self.fit_info["ierr"] = ierr
        if ierr not in [1, 2, 3, 4]:
            # apparently setting a higher stacklevel is better, see
            # https://docs.astral.sh/ruff/rules/no-explicit-stacklevel/
            warnings.warn(
                "The fit may be unsuccessful; check "
                "fit_info['message'] for more information.",
                AstropyUserWarning,
                stacklevel=2,
            )

        # now try to compute the true covariance matrix
        if (len(args[-1]) > len(init_values)) and cov_x is not None:
            sum_sqrs = np.sum(self.objective_function(fitparams, *farg) ** 2)
            dof = len(args[-1]) - len(init_values)
            self.fit_info["param_cov"] = cov_x * sum_sqrs / dof
        else:
            self.fit_info["param_cov"] = None

        return model_copy


class TransitModelOptions(BaseModel):
    bin_size: float = 5.0
    keep_transit_time_fixed: bool = True
    transit_time_range: float = 60.0
    keep_radius_planet_fixed: bool = False
    keep_radius_orbit_fixed: bool = False
    fit_airmass: bool = False
    fit_width: bool = False
    fit_spp: bool = False


class TransitModelFit:
    """
    Transit model fits to observed light curves.

    Parameters
    ----------
    batman_params : batman.TransitParams
        Parameters for the batman transit model. If not provided, the
        default parameters will be used.

    Attributes
    ----------

    BIC : float
        Bayesian Information Criterion for the fit. This is calculated
        after the fit is performed.

    n_fit_parameters : int
        Number of parameters that were fit. This is calculated after the
        fit is performed.

    width : array-like
        Width of the star in pixels at each time. Must be set before fitting.
    """

    def __init__(self):
        self._batman_params = batman.TransitParams()
        self._set_default_batman_params()
        self._times = None
        self._airmass = None
        self._spp = None
        self._width = None
        self._fitter = VariableArgsFitter()
        self._model = None
        self._batman_mod_for_fit = None
        self.weights = None
        self._detrend_parameters = set()
        self._all_detrend_params = ["airmass", "width", "spp"]

    def _check_consistent_lengths(self, proposed_value):
        """
        Check that the proposed value has a length consistent with the
        other independent variables. Consistent means same length as others
        or the others are ``None``.
        """
        if proposed_value is None:
            return True

        new_length = len(proposed_value)
        for independent_var in [self._times, self._airmass, self._spp, self._width]:
            if independent_var is None:
                continue
            elif len(independent_var) != new_length:
                return False
        else:
            # All the lengths were good
            return True

    @property
    def times(self):
        """
        times : array-like
            Times at which the light curve is observed. Must be set before
            fitting.
        """
        return self._times

    @times.setter
    def times(self, value):
        if not self._check_consistent_lengths(value):
            raise ValueError(
                "Length of times not consistent with " "other independent variables."
            )
        self._times = value

        try:
            if self._batman_mod_for_fit is None:
                self._set_up_batman_model_for_fitting()
        except ValueError:
            pass

    @property
    def airmass(self):
        """
        airmass : array-like
            Airmass at each time. Must be set before fitting.
        """
        return self._airmass

    @airmass.setter
    def airmass(self, value):
        if not self._check_consistent_lengths(value):
            raise ValueError(
                "Length of airmass not consistent with " "other independent variables."
            )
        self._airmass = value

        if value is not None:
            self._detrend_parameters.add("airmass")
        else:
            self._detrend_parameters.discard("airmass")

    @property
    def width(self):
        """
        weights : array-like
            Weights to use for fitting. If not provided, all weights are
            set to 1.
        """
        return self._width

    @width.setter
    def width(self, value):
        if not self._check_consistent_lengths(value):
            raise ValueError(
                "Length of width not consistent with " "other independent variables."
            )
        self._width = value

        if value is not None:
            self._detrend_parameters.add("width")
        else:
            self._detrend_parameters.discard("width")

    @property
    def spp(self):
        """
        spp : array-like
            Sky per pixel at each time. Must be set before fitting.
        """
        return self._spp

    @spp.setter
    def spp(self, value):
        if not self._check_consistent_lengths(value):
            raise ValueError(
                "Length of spp not consistent with " "other independent variables."
            )
        self._spp = value

        if value is not None:
            self._detrend_parameters.add("spp")
        else:
            self._detrend_parameters.discard("spp")

    @property
    def data(self):
        """
        data : array-like
            Observed fluxes. Must be set before fitting.
        """
        return self._data

    @data.setter
    def data(self, value):
        if not self._check_consistent_lengths(value):
            raise ValueError(
                "Length of data not consistent with " "independent variables."
            )
        self._data = value

    @property
    def model(self):
        """
        model : `astropy.modeling.Model`
            The model used for fitting. This is a combination of the batman
            transit model and any other trends that are included in the fit.
            This is set up when the ``setup_model`` method is called.
        """
        return self._model

    def _set_default_batman_params(self):
        """
        Initialize batman parameters to some not nonsensical values. These
        will need to be adjusted before fitting.
        """
        # time of inferior conjunction --> THE MIDPOINT of the transit
        self._batman_params.t0 = 59028.723

        # orbital period
        self._batman_params.per = 5.72

        # planet radius (in units of stellar radii)  --> DETERMINES DEPTH
        # (affects duration)
        self._batman_params.rp = 0.035

        # semi-major axis (in units of stellar radii) --> DETERMINES DURATION
        self._batman_params.a = 12.2

        # orbital inclination (in degrees)
        self._batman_params.inc = 90.0

        # eccentricity
        self._batman_params.ecc = 0.0

        # longitude of periastron (in degrees)
        self._batman_params.w = 90.0

        # limb darkening model
        self._batman_params.limb_dark = "quadratic"

        # limb darkening coefficients [u1, u2]
        self._batman_params.u = [0.3, 0.3]

    def _set_up_batman_model_for_fitting(self):
        if self._times is None:
            raise ValueError("times need to be set before setting up " "transit model.")
        self._batman_mod_for_fit = batman.TransitModel(self._batman_params, self.times)

    def _setup_transit_model(self):
        """
        Creates transit astropy model with exoplanet flux and other trends.
        Called when the necessary information is present for setting up the
        model.
        """

        def transit_model_with_trends(
            time,  # noqa: ARG001  (model needs this argument as independent variable)
            airmass,
            width,
            sky_per_pix,
            t0=0.0,
            period=1.0,
            rp=0.1,
            a=10.0,
            inclination=90.0,
            eccentricity=0.0,
            limb_u1=0.3,
            limb_u2=0.3,
            airmass_trend=0.0,
            width_trend=0.0,
            spp_trend=0.0,
        ):
            self._batman_params.t0 = t0
            self._batman_params.per = period
            self._batman_params.rp = rp
            self._batman_params.a = a
            self._batman_params.inc = inclination
            self._batman_params.ecc = eccentricity
            self._batman_params.u = [limb_u1, limb_u2]

            flux = self._batman_mod_for_fit.light_curve(self._batman_params)
            flux += airmass_trend * (airmass)
            flux += width_trend * width
            flux += spp_trend * sky_per_pix
            return flux

        ModelClass = custom_model(transit_model_with_trends)

        self._model = ModelClass()

        # Set up some defaults for what is fixed
        self._model.period.fixed = True
        self._model.eccentricity.fixed = True
        self._model.limb_u1.fixed = True
        self._model.limb_u2.fixed = True

        # Set some default bounds for inclination, in degrees.
        self._model.inclination.bounds = (80, 90)

        # Planet radius cannot be too small or too big
        self._model.rp.bounds = (0.01, 0.5)

    def setup_model(
        self,
        binned_data=None,
        t0=0,
        depth=0,
        duration=0,
        period=1,
        inclination=90,
        airmass_trend=0.0,
        width_trend=0.0,
        spp_trend=0.0,
        model_options=None,
    ):
        """
        Configure a transit model for fitting. The ``duration`` and ``depth``
        are used to estimate underlying fit parameters; they are not
        themselves fit parameters.

        Parameters
        ----------
        t0 : float
            Time of the center of the transit. Can be in any units
            but should be consistent with the units for the ``period``
            and for the times used for fitting.

        depth : float
            Depth of the transit in parts per thousand.

        duration : float
            Duration of the transit,in the same units as ``t0`` and
            ``period``.

        period : float
            Period of the planet. Should be in the same units as ``t0``
            and times used for fitting.

        inclination : float
            Inclination of the orbit, in degrees.

        airmass_trend : float, optional
            Coefficient for a linear trend in airmass.

        width_trend : float
            Coefficient for a linear trend in stellar width.

        spp_trend : float
            Coefficient for a linear trend in sky per pixel.

        options : TransitModelOptions, optional
            Options for the transit model fit.

        Returns
        -------
        None
            Sets values for the model parameters.
        """
        if binned_data:
            self.times = (
                np.array(
                    (
                        binned_data["time_bin_start"] + binned_data["time_bin_size"] / 2
                    ).value
                )
                - 2400000
            )
            self.data = binned_data["normalized_flux"].value
            self.weights = 1 / (binned_data["normalized_flux_error"].value)
            self.airmass = np.array(binned_data["airmass"])
            self.width = np.array(binned_data["width"])
            self.spp = np.array(binned_data["sky_per_pix_avg"])
        self._setup_transit_model()

        # rp is related to depth in a straightforward way
        self._model.rp.value = self._batman_params.rp = np.sqrt(depth / 1000)

        # The estimate below assumes a circular orbit and inclination of
        # 90 degrees (edge on). This should be fine as a starting point
        # for the fit.
        #
        # See Kipping, eq. 16 at
        # https://doi.org/10.1111/j.1365-2966.2010.16894.x
        estimated_a = 1 / np.sin(duration * np.pi / period)
        self._model.a.value = self._batman_params.a = estimated_a

        self._model.period.value = self._batman_params.per = period
        self._model.inclination.value = self._batman_params.inc = inclination
        self._model.t0.value = self._batman_params.t0 = t0
        self._model.airmass_trend.value = airmass_trend
        self._model.width_trend.value = width_trend
        self._model.spp_trend.value = spp_trend

        try:
            if self._batman_mod_for_fit is None:
                self._set_up_batman_model_for_fitting()
        except ValueError:
            pass

        if model_options is not None:
            # Setup the model more 🙄
            self.model.t0.bounds = [
                t0 - (model_options.transit_time_range * u.min).to("day").value / 2,
                t0 + (model_options.transit_time_range * u.min).to("day").value / 2,
            ]
            self.model.t0.fixed = model_options.keep_transit_time_fixed
            self.model.a.fixed = model_options.keep_radius_orbit_fixed
            self.model.rp.fixed = model_options.keep_radius_planet_fixed

            self.model.spp_trend.fixed = not model_options.fit_spp
            self.model.airmass_trend.fixed = not model_options.fit_airmass
            self.model.width_trend.fixed = not model_options.fit_width

    def fit(self):
        """
        Perform a fit and update the model with best-fit values.
        """
        # Maybe do some correctness check of the model before starting.
        if self.times is None:
            raise ValueError("The times must be set before trying " "to fit.")
        if self._model is None:
            raise ValueError("Run setup_model() before trying to fit.")

        if self._batman_mod_for_fit is None:
            self._set_up_batman_model_for_fitting()

        # Check whether any data bits are None and fix those
        # parameters.
        original_values = {}

        if self.spp is None:
            original_values["spp_trend"] = self.model.spp_trend.fixed
            self.model.spp_trend = 0
            self.model.spp_trend.fixed = True
            spp = np.zeros_like(self.times)
        else:
            spp = self.spp

        if self.airmass is None:
            original_values["airmass_trend"] = self.model.airmass_trend.fixed
            self.model.airmass_trend = 0
            self.model.airmass_trend.fixed = True
            airmass = np.zeros_like(self.times)
        else:
            airmass = self.airmass

        if self.width is None:
            original_values["width_trend"] = self.model.width_trend.fixed
            self.model.width_trend = 0
            self.model.width_trend.fixed = True
            width = np.zeros_like(self.times)
        else:
            width = self.width

        # Do the fitting

        new_model = self._fitter(
            self.model, self.times, airmass, width, spp, self.data, weights=self.weights
        )

        # Update the model (might not be necessary but can't hurt)
        self._model = new_model
        self._actual_fixed_params = self._model.fixed

        # reset parameters to their original values
        for k, v in original_values.items():
            param = getattr(self.model, k)
            param.fixed = v

    def _detrend(self, model, detrend_by):
        if detrend_by == "all":
            detrend_by = [
                p for p in self._all_detrend_params if p in self._detrend_parameters
            ]
        elif isinstance(detrend_by, str):
            detrend_by = [detrend_by]

        detrended = model.copy()
        for trend in detrend_by:
            detrended = detrended - (
                getattr(self.model, f"{trend}_trend") * getattr(self, trend)
            )

        return detrended

    def data_light_curve(self, data=None, detrend_by=None):
        """
        Function to return data light curve, optionally detrended by one or
        more parameters.

        Parameters
        ----------

        data : array-like, optional
            Data to use for calculating the light curve. If not provided,
            the data used for fitting will be used.

        detrend_by : str or list of str
            Parameter(s) to detrend by. If ``None``, no detrending is
            done. If ``'all'``, all parameters that are set will be
            used for detrending.

        Returns
        -------

        data : array-like
            Data light curve, detrended if requested.
        """
        data = data if data is not None else self.data

        if detrend_by is not None:
            data = self._detrend(data, detrend_by)

        return data

    def model_light_curve(self, at_times=None, detrend_by=None):
        """
        Calculate the light curve corresponding to the model, optionally
        detrended by one or more parameters.

        Parameters
        ----------
        at_times : array-like
            Times at which to calculate the model. If not provided, the
            times used for fitting will be used.

        detrend_by : str or list of str
            Parameter(s) to detrend by. If ``None``, no detrending is
            done. If ``'all'``, all parameters that are set will be
            used for detrending.

        Returns
        -------
        model : array-like
            Model light curve.
        """
        zeros = np.zeros_like(self.times)
        airmass = self.airmass if self.airmass is not None else zeros
        width = self.width if self.width is not None else zeros
        spp = self.spp if self.spp is not None else zeros

        if at_times is not None:
            # temporarily reset the batman model to the new times,
            # then restore it.
            original_model = self._batman_mod_for_fit

            self._batman_mod_for_fit = batman.TransitModel(
                self._batman_params, at_times
            )
            model = self.model(at_times, airmass, width, spp)
            self._batman_mod_for_fit = original_model
        else:
            model = self.model(self.times, airmass, width, spp)

        if detrend_by is not None:
            model = self._detrend(model, detrend_by)

        return model

    @property
    def n_fit_parameters(self):
        return sum(not v for k, v in self._actual_fixed_params.items())

    @property
    def BIC(self):
        residual = self.data - self.model_light_curve()
        chi_sq = ((residual * self.weights) ** 2).sum()
        BIC = chi_sq + self.n_fit_parameters * np.log(len(self.data))
        return BIC


# example use

# self.model.eccentricity.fixed = True
# self.model.spp_trend = 0
# self.model.spp_trend.fixed = True

# self.model.t0 = ...
# self.model.rp = ...
# self.fit(time, airmass, width, sky_per_pix)
# self.model.rp
# self.model.rp.error

# my_model = TransitModelFit()
# my_model.times = ...
# my_model.airmass = ...

# my_model.setup_model()
# my_model.model.t0 = ...
# my_model.model.rp = ...
# my_model.model.period = 123
# my_model.model.period.fixed = True

# my_model.data = ...

# fit_model = my_model.fit()

# my_model.model_light_curve(detrend_by=['airmass', 'spp'])

# my_model.model_light_curve(at_times=np.linspace(start - 0.1, end + 0.1, num=1000),
#                            detrend_by=['airmass'])

# my_model.model
