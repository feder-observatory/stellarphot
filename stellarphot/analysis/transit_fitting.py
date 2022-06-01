import warnings

import numpy as np

from astropy.modeling.models import Polynomial1D, Gaussian1D, custom_model
from astropy.modeling.fitting import (LevMarLSQFitter,
                                      _validate_model,
                                      _convert_input)

# Functions blow changed from private to public in astropy 5
try:
    from astropy.modeling.fitting import (
        fitter_to_model_params,
        model_to_fit_params,
    )
except ImportError:
    from astropy.modeling.fitting import (
        _fitter_to_model_params as fitter_to_model_params,
        _model_to_fit_params as model_to_fit_params,
    )

from astropy.utils.exceptions import AstropyUserWarning

try:
    import batman
except ImportError:
    ImportError('You must install the batman exoplanet package. Try:\n'
                'conda install batman-package\n'
                'or\n'
                'pip install batman-package')


class VariableArgsFitter(LevMarLSQFitter):
    """
    Allow fitting of functions with arbitrary number of positional
    parameters.
    """
    def __init__(self):
        super().__init__()

    # This is a straight copy-paste from the LevMarLSQFitter __call__.
    # The only modification is to allow any number of arguments.
    def __call__(self, model, *args, weights=None,
                 maxiter=100, acc=1e-7,
                 epsilon=1.4901161193847656e-08, estimate_jacobian=False):
        from scipy import optimize

        model_copy = _validate_model(model, self.supported_constraints)
        farg = (model_copy, weights, ) + args
        if model_copy.fit_deriv is None or estimate_jacobian:
            dfunc = None
        else:
            dfunc = self._wrap_deriv
        init_values = model_to_fit_params(model_copy)
        fitparams, cov_x, dinfo, mess, ierr = optimize.leastsq(
            self.objective_function, init_values, args=farg, Dfun=dfunc,
            col_deriv=model_copy.col_fit_deriv, maxfev=maxiter, epsfcn=epsilon,
            xtol=acc, full_output=True)
        fitter_to_model_params(model_copy, fitparams)
        self.fit_info.update(dinfo)
        self.fit_info['cov_x'] = cov_x
        self.fit_info['message'] = mess
        self.fit_info['ierr'] = ierr
        if ierr not in [1, 2, 3, 4]:
            warnings.warn("The fit may be unsuccessful; check "
                          "fit_info['message'] for more information.",
                          AstropyUserWarning)

        # now try to compute the true covariance matrix
        if (len(args[-1]) > len(init_values)) and cov_x is not None:
            sum_sqrs = np.sum(self.objective_function(fitparams, *farg)**2)
            dof = len(args[-1]) - len(init_values)
            self.fit_info['param_cov'] = cov_x * sum_sqrs / dof
        else:
            self.fit_info['param_cov'] = None

        return model_copy


class TransitModelFit:
    def __init__(self, batman_params=None):
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
        self._all_detrend_params = ['airmass', 'width', 'spp']

    def _check_consistent_lengths(self, proposed_value):
        """
        Check that the proposed value has a length consistent with the
        other independent variables. Consistent means same length as others
        or the others are ``None``.
        """
        if proposed_value is None:
            return True

        new_length = len(proposed_value)
        for independent_var in [self._times, self._airmass,
                                self._spp, self._width]:
            if independent_var is None:
                continue
            elif len(independent_var) != new_length:
                return False
        else:
            # All the lengths were good
            return True

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, value):
        if not self._check_consistent_lengths(value):
            raise ValueError('Length of times not consistent with '
                             'other independent variables.')
        self._times = value

        try:
            if self._batman_mod_for_fit is None:
                self._set_up_batman_model_for_fitting()
        except ValueError:
            pass

    @property
    def airmass(self):
        return self._airmass

    @airmass.setter
    def airmass(self, value):
        if not self._check_consistent_lengths(value):
            raise ValueError('Length of airmass not consistent with '
                             'other independent variables.')
        self._airmass = value

        if value is not None:
            self._detrend_parameters.add('airmass')
        else:
            self._detrend_parameters.discard('airmass')

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if not self._check_consistent_lengths(value):
            raise ValueError('Length of width not consistent with '
                             'other independent variables.')
        self._width = value

        if value is not None:
            self._detrend_parameters.add('width')
        else:
            self._detrend_parameters.discard('width')

    @property
    def spp(self):
        return self._spp

    @spp.setter
    def spp(self, value):
        if not self._check_consistent_lengths(value):
            raise ValueError('Length of spp not consistent with '
                             'other independent variables.')
        self._spp = value

        if value is not None:
            self._detrend_parameters.add('spp')
        else:
            self._detrend_parameters.discard('spp')

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not self._check_consistent_lengths(value):
            raise ValueError('Length of data not consistent with '
                             'independent variables.')
        self._data = value

    @property
    def model(self):
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

        # planet radius (in units of stellar radii)  --> DETERMINES DEPTH (affects duration)
        self._batman_params.rp = 0.035

        # semi-major axis (in units of stellar radii) --> DETERMINES DURATION
        self._batman_params.a = 12.2

        # orbital inclination (in degrees)
        self._batman_params.inc = 90.

        # eccentricity
        self._batman_params.ecc = 0.

        # longitude of periastron (in degrees)
        self._batman_params.w = 90.

        # limb darkening model
        self._batman_params.limb_dark = "quadratic"

        # limb darkening coefficients [u1, u2]
        self._batman_params.u = [0.3, 0.3]

    def _set_up_batman_model_for_fitting(self):
        if self._times is None:
            raise ValueError('times need to be set before setting up '
                             'transit model.')
        self._batman_mod_for_fit = \
            batman.TransitModel(self._batman_params,
                                self.times)

    def _setup_transit_model(self):
        """
        Creates transit astropy model with exoplanet flux and other trends.
        Called when the necessary information is present for setting up the
        model.
        """

        def transit_model_with_trends(time, airmass, width, sky_per_pix,
                                      t0=0.0, period=1.0, rp=0.1, a=10.0,
                                      inclination=90.0, eccentricity=0.0,
                                      limb_u1=0.3, limb_u2=0.3,
                                      airmass_trend=0.0, width_trend=0.0,
                                      spp_trend=0.0):
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

    def setup_model(self, t0=0, depth=0, duration=0,
                    period=1, inclination=90,
                    airmass_trend=0.0,
                    width_trend=0.0,
                    spp_trend=0.0):
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
        """
        self._setup_transit_model()

        # rp is related to depth in a straighforward way
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

    def fit(self):
        """
        Perform a fit and update the model with best-fit values.
        """
        # Maybe do some correctness check of the model before starting.
        if self.times is None:
            raise ValueError('The times must be set before trying '
                             'to fit.')
        if self._model is None:
            raise ValueError('Run setup_model() before trying to fit.')

        if self._batman_mod_for_fit is None:
            self._set_up_batman_model_for_fitting()

        # Check whether any data bits are None and fix those
        # parameters.
        original_values = {}

        if self.spp is None:
            original_values['spp_trend'] = \
                self.model.spp_trend.fixed
            self.model.spp_trend = 0
            self.model.spp_trend.fixed = True
            spp = np.zeros_like(self.times)
        else:
            spp = self.spp

        if self.airmass is None:
            original_values['airmass_trend'] = \
                self.model.airmass_trend.fixed
            self.model.airmass_trend = 0
            self.model.airmass_trend.fixed = True
            airmass = np.zeros_like(self.times)
        else:
            airmass = self.airmass

        if self.width is None:
            original_values['width_trend'] = \
                self.model.width_trend.fixed
            self.model.width_trend = 0
            self.model.width_trend.fixed = True
            width = np.zeros_like(self.times)
        else:
            width = self.width

        # Do the fitting

        new_model = self._fitter(self.model,
                                 self.times,
                                 airmass,
                                 width,
                                 spp,
                                 self.data,
                                 weights=self.weights)

        # Update the model (might not be necessary but can't hurt)
        self._model = new_model
        self._actual_fixed_params = self._model.fixed

        # reset parameters to their original values
        for k, v in original_values.items():
            param = getattr(self.model, k)
            param.fixed = v

    def _detrend(self, model, detrend_by):
        if detrend_by == 'all':
            detrend_by = [p for p in self._all_detrend_params
                          if p in self._detrend_parameters]
        elif isinstance(detrend_by, str):
            detrend_by = [detrend_by]

        detrended = model.copy()
        for trend in detrend_by:
            detrended = detrended - (getattr(self.model, f'{trend}_trend')
                                     * getattr(self, trend))

        return detrended

    def data_light_curve(self, data=None, detrend_by=None):
        data = data if data is not None else self.data

        if detrend_by is not None:
            data = self._detrend(data, detrend_by)

        return data

    def model_light_curve(self, at_times=None, detrend_by=None):
        """
        Calculate the light curve corresponding to the model, optionally
        detrended by one or more parameters.
        """
        zeros = np.zeros_like(self.times)
        airmass = self.airmass if self.airmass is not None else zeros
        width = self.width if self.width is not None else zeros
        spp = self.spp if self.spp is not None else zeros

        if at_times is not None:
            # temporarily reset the batman model to the new times,
            # then restore it.
            original_model = self._batman_mod_for_fit

            self._batman_mod_for_fit = batman.TransitModel(self._batman_params,
                                                           at_times)
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
        chi_sq = ((residual * self.weights)**2).sum()
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
