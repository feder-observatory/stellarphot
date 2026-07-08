import itertools

import numpy as np
from astropy import units as u
from astropy.table import Table
from pydantic import BaseModel

try:
    from pytransit import RoadRunnerModel
except ImportError:  # pragma: no cover
    # pytransit is an optional dependency (the ``exoplanet`` extra). Importing
    # this module must still succeed without it so the rest of stellarphot can
    # be imported; instantiating TransitModelFit without pytransit raises a
    # clear error (see __init__).
    RoadRunnerModel = None

try:
    import lmfit
except ImportError:  # pragma: no cover
    # lmfit is an optional dependency (the ``exoplanet`` extra), guarded the
    # same way as pytransit.
    lmfit = None

_PYTRANSIT_INSTALL_MESSAGE = (
    "You must install pytransit to use TransitModelFit. Try:\n"
    "  conda install -c conda-forge pytransit\n"
    "or\n"
    "  pip install pytransit"
)

_LMFIT_INSTALL_MESSAGE = (
    "You must install lmfit to use TransitModelFit. Try:\n"
    "  conda install -c conda-forge lmfit\n"
    "or\n"
    "  pip install lmfit"
)

__all__ = ["TransitModelOptions", "TransitModelFit"]


def _default_params():
    """
    Default parameter set for a transit fit.

    Trend fitting is opt-in: the trend coefficients start with
    ``vary=False`` and are only fit if the user (or
    `~stellarphot.transit_fitting.TransitModelOptions`) turns them on.
    """
    params = lmfit.Parameters()
    params.add("t0", value=0.0, vary=True)
    params.add("period", value=1.0, vary=False)
    params.add("rp", value=0.1, vary=True, min=0.01, max=0.5)
    params.add("a", value=10.0, vary=True, min=1.0)
    params.add("inclination", value=90.0, vary=True, min=50, max=90)
    params.add("limb_u1", value=0.3, vary=False)
    params.add("limb_u2", value=0.3, vary=False)
    params.add("airmass_trend", value=0.0, vary=False)
    params.add("width_trend", value=0.0, vary=False)
    params.add("spp_trend", value=0.0, vary=False)
    return params


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

    The underlying transit light curve is computed with pytransit's
    `RoadRunnerModel
    <https://pytransit.readthedocs.io/en/latest/notebooks/models/roadrunner/roadrunner_model_example_1.html>`_
    using a quadratic limb-darkening law and a circular orbit. The fit is
    performed with `lmfit <https://lmfit.github.io/lmfit-py/>`_.

    Attributes
    ----------
    params : `lmfit.Parameters`
        The transit model parameters: ``t0``, ``period``, ``rp`` (planet
        radius in stellar radii), ``a`` (orbital radius in stellar radii),
        ``inclination`` (degrees), ``limb_u1``/``limb_u2`` (quadratic
        limb-darkening coefficients) and the linear trend coefficients
        ``airmass_trend``, ``width_trend`` and ``spp_trend``. Read or set
        values, bounds and whether a parameter is fit through the standard
        lmfit attributes, e.g. ``mod.params["rp"].value``,
        ``mod.params["t0"].vary``, ``mod.params["a"].min`` and, after a
        fit, ``mod.params["rp"].stderr``.

    fit_result : `lmfit.minimizer.MinimizerResult` or None
        Result of the most recent call to ``fit``, including fit statistics
        like ``bic`` and ``nvarys``. ``None`` until ``fit`` has been run.

    times, airmass, width, spp, data, weights : array-like or None
        Independent variables and data for the fit; see the property
        docstrings.

    Notes
    -----
    The orbit is always circular; there is no eccentricity parameter.

    Trend fitting is opt-in: the trend coefficients default to
    ``vary=False`` even when a covariate (airmass, width, sky per pixel) is
    set. Enable them via `TransitModelOptions`, directly (e.g.
    ``mod.params["airmass_trend"].vary = True``) or with
    ``compare_detrend_options(apply_best=True)``.

    Examples
    --------
    Fit a transit, detrending against airmass::

        mod = TransitModelFit()
        mod.setup_model(t0=2455001.5, depth=12.1, duration=0.15, period=3.5)
        mod.times = times
        mod.data = fluxes
        mod.weights = 1 / flux_errors
        mod.airmass = airmass
        mod.params["airmass_trend"].vary = True

        result = mod.fit()

        print(mod.params["rp"].value, mod.params["rp"].stderr)
        print(result.bic)

        # Which detrending parameters does the BIC favor?
        bic_table = mod.compare_detrend_options()
    """

    def __init__(self):
        if RoadRunnerModel is None:
            raise ImportError(_PYTRANSIT_INSTALL_MESSAGE)
        if lmfit is None:
            raise ImportError(_LMFIT_INSTALL_MESSAGE)

        # pytransit's RoadRunnerModel with the quadratic limb-darkening law is
        # parameterized at evaluation time via ``evaluate(...)``, so no separate
        # parameter container is needed. The time array is supplied later via
        # ``set_data`` (see the ``times`` setter).
        self._transit_model = RoadRunnerModel("quadratic")
        self._times = None
        self._airmass = None
        self._spp = None
        self._width = None
        self._data = None
        self.weights = None
        self.params = _default_params()
        self.fit_result = None
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
                "Length of times not consistent with other independent variables."
            )
        self._times = value

        if value is not None:
            self._transit_model.set_data(np.asarray(value, dtype=float))

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
                "Length of airmass not consistent with other independent variables."
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
                "Length of width not consistent with other independent variables."
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
                "Length of spp not consistent with other independent variables."
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
                "Length of data not consistent with independent variables."
            )
        self._data = value

    def _model_flux(self, params, airmass, width, spp):
        """
        Evaluate the model flux (transit plus additive linear trends) for the
        given parameters at the times currently set on the pytransit model.
        """
        p = params.valuesdict()
        flux = self._transit_model.evaluate(
            k=p["rp"],
            ldc=[p["limb_u1"], p["limb_u2"]],
            t0=p["t0"],
            p=p["period"],
            a=p["a"],
            i=np.radians(p["inclination"]),
        )
        flux += p["airmass_trend"] * airmass
        flux += p["width_trend"] * width
        flux += p["spp_trend"] * spp
        return flux

    def _residual(self, params, airmass, width, spp, weights):
        return (self._model_flux(params, airmass, width, spp) - self.data) * weights

    def _run_fit(self, params):
        """
        Run an lmfit minimization with the given parameters, returning the
        `lmfit.minimizer.MinimizerResult`.

        The fit runs on a copy of ``params``, so neither the input nor any
        instance state is modified. For each covariate that is not set, the
        corresponding trend parameter is forced to zero (and not varied) on
        that copy and an all-zero covariate is substituted.
        """
        params = params.copy()

        covariates = {}
        for name in self._all_detrend_params:
            value = getattr(self, name)
            if value is None:
                params[f"{name}_trend"].set(value=0.0, vary=False)
                covariates[name] = np.zeros(len(self.times))
            else:
                covariates[name] = np.asarray(value, dtype=float)

        weights = self.weights if self.weights is not None else 1.0

        # method="least_squares" is scipy's Trust Region Reflective, which
        # handles a starting value that sits exactly on a bound (the default
        # inclination of 90 degrees is on its upper bound); MINPACK leastsq
        # does not.
        return lmfit.minimize(
            self._residual,
            params,
            method="least_squares",
            args=(
                covariates["airmass"],
                covariates["width"],
                covariates["spp"],
                weights,
            ),
        )

    def setup_model(
        self,
        *,
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
        themselves fit parameters. Any previous parameter settings are reset
        to their defaults before the new values are applied.

        All arguments are keyword-only.

        Parameters
        ----------
        binned_data : `astropy.timeseries.BinnedTimeSeries`, optional
            Binned time series to load the times, data, weights and
            covariates from.

        t0 : float
            Time of the center of the transit. Can be in any units
            but should be consistent with the units for the ``period``
            and for the times used for fitting.

        depth : float
            Depth of the transit in parts per thousand. If zero (the
            default), the planet radius ``rp`` is left at its default
            starting value instead of being estimated.

        duration : float
            Duration of the transit, in the same units as ``t0`` and
            ``period``. If zero (the default), the orbital radius ``a`` is
            left at its default starting value instead of being estimated.
            Must be smaller than the ``period``.

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

        model_options : TransitModelOptions, optional
            Options for the transit model fit, mapped onto the ``vary``
            flags (and, for ``t0``, the bounds) of the parameters.

        Returns
        -------
        None
            Sets values for the model parameters.

        Raises
        ------
        ValueError
            If ``duration`` is not smaller than ``period``.
        """
        if duration >= period:
            raise ValueError(
                f"The transit duration ({duration}) must be smaller than "
                f"the period ({period})."
            )

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

        self.params = _default_params()

        # rp is related to depth in a straightforward way; a nonpositive
        # depth means "no estimate," leaving rp at its default. Seeding rp
        # at zero would in any case start the fit below rp's lower bound.
        if depth > 0:
            self.params["rp"].value = np.sqrt(depth / 1000)

        # The estimate below assumes a circular orbit and inclination of
        # 90 degrees (edge on). This should be fine as a starting point
        # for the fit. A nonpositive duration means "no estimate," leaving
        # a at its default.
        #
        # See Kipping, eq. 16 at
        # https://doi.org/10.1111/j.1365-2966.2010.16894.x
        if duration > 0:
            self.params["a"].value = 1 / np.sin(duration * np.pi / period)

        self.params["period"].value = period
        self.params["inclination"].value = inclination
        self.params["t0"].value = t0
        self.params["airmass_trend"].value = airmass_trend
        self.params["width_trend"].value = width_trend
        self.params["spp_trend"].value = spp_trend

        if model_options is not None:
            half_range = (model_options.transit_time_range * u.min).to("day").value / 2
            self.params["t0"].min = t0 - half_range
            self.params["t0"].max = t0 + half_range
            self.params["t0"].vary = not model_options.keep_transit_time_fixed

            self.params["a"].vary = not model_options.keep_radius_orbit_fixed
            self.params["rp"].vary = not model_options.keep_radius_planet_fixed

            self.params["airmass_trend"].vary = model_options.fit_airmass
            self.params["width_trend"].vary = model_options.fit_width
            self.params["spp_trend"].vary = model_options.fit_spp

    def fit(self):
        """
        Fit the model to the data and update ``params`` with the best-fit
        values and uncertainties.

        Returns
        -------
        `lmfit.minimizer.MinimizerResult`
            The full fit result, also stored as ``fit_result``. Includes
            fit statistics like ``bic``.

        Raises
        ------
        ValueError
            If the times or the data have not been set. If the fit itself
            raises, ``params`` and ``fit_result`` are left untouched.
        """
        if self.times is None:
            raise ValueError("The times must be set before trying to fit.")
        if self.data is None:
            raise ValueError("The data must be set before trying to fit.")

        result = self._run_fit(self.params)

        self.fit_result = result

        # Copy the best-fit values and uncertainties back into the user's
        # parameters without touching vary/min/max, so user choices (e.g. a
        # trend enabled while its covariate is unset) survive the fit.
        for name, param in result.params.items():
            self.params[name].value = param.value
            self.params[name].stderr = param.stderr

        return result

    def compare_detrend_options(self, apply_best=False):
        """
        Compare, via the Bayesian Information Criterion, fits with every
        on/off combination of the trend parameters whose covariate is set.

        Each candidate fit runs on a copy of ``params`` — with the trends in
        the combination varying, and the others fixed at zero — so this
        method does not change the state of the model.

        Parameters
        ----------
        apply_best : bool, optional
            If True, apply the lowest-BIC combination to ``params`` (each
            trend in the winning combination is varied, the others are
            zeroed and fixed) and run ``fit``.

        Returns
        -------
        `astropy.table.Table`
            One row per combination with a bool column per available trend
            (``airmass``, ``width``, ``spp``) and a ``BIC`` column, sorted
            by ascending BIC (best first).
        """
        if self.times is None:
            raise ValueError("The times must be set before trying to fit.")
        if self.data is None:
            raise ValueError("The data must be set before trying to fit.")

        available = [
            name
            for name in self._all_detrend_params
            if name in self._detrend_parameters
        ]

        rows = []
        for combo in itertools.product([False, True], repeat=len(available)):
            params = self.params.copy()
            for name, enabled in zip(available, combo, strict=True):
                if enabled:
                    params[f"{name}_trend"].vary = True
                else:
                    params[f"{name}_trend"].set(value=0.0, vary=False)

            result = self._run_fit(params)

            row = dict(zip(available, combo, strict=True))
            row["BIC"] = result.bic
            rows.append(row)

        comparison = Table(rows=rows)
        comparison.sort("BIC")

        if apply_best:
            best = comparison[0]
            for name in available:
                if best[name]:
                    self.params[f"{name}_trend"].vary = True
                else:
                    self.params[f"{name}_trend"].set(value=0.0, vary=False)
            self.fit()

        return comparison

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
                self.params[f"{trend}_trend"].value * getattr(self, trend)
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
            times used for fitting will be used. Because the airmass/width/spp
            trend covariates are reused as-is, ``at_times`` must have the same
            length as the model's times; a mismatched length raises
            ``ValueError``.

        detrend_by : str or list of str
            Parameter(s) to detrend by. If ``None``, no detrending is
            done. If ``'all'``, all parameters that are set will be
            used for detrending.

        Returns
        -------
        model : array-like
            Model light curve.
        """
        if self.times is None:
            raise ValueError(
                "The times must be set before computing a model light curve."
            )

        zeros = np.zeros(len(self.times))
        airmass = self.airmass if self.airmass is not None else zeros
        width = self.width if self.width is not None else zeros
        spp = self.spp if self.spp is not None else zeros

        if at_times is not None:
            at_times = np.asarray(at_times, dtype=float)
            # The trend covariates (airmass/width/spp) are the ones supplied for
            # the model's own times, so at_times must line up with them. Check
            # before repointing the pytransit model so a bad call cannot leave
            # the instance in a corrupted state.
            if len(at_times) != len(self.times):
                raise ValueError(
                    "at_times must have the same length as the model times "
                    f"({len(self.times)}) so the airmass/width/spp trend "
                    f"covariates line up; got length {len(at_times)}."
                )
            # Temporarily point the transit model at the new times, restoring
            # the original times even if evaluation fails.
            self._transit_model.set_data(at_times)
            try:
                model = self._model_flux(self.params, airmass, width, spp)
            finally:
                self._transit_model.set_data(np.asarray(self.times, dtype=float))
        else:
            model = self._model_flux(self.params, airmass, width, spp)

        if detrend_by is not None:
            model = self._detrend(model, detrend_by)

        return model
