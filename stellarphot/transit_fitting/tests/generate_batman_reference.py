"""
One-off generator for the batman reference light curves used by the
pytransit regression test in ``test_transit_model_fit.py``.

stellarphot's transit fitting used to rely on ``batman-package`` and now uses
``pytransit``. To make sure pytransit reproduces batman's results without
keeping batman as a perpetual test dependency, we use batman *once* -- here --
to generate a small fixture of reference light curves that is committed to the
repository. The regression test then only needs pytransit.

Run this manually (with ``batman-package`` installed) to regenerate the fixture:

    python stellarphot/transit_fitting/tests/generate_batman_reference.py

It is intentionally guarded by ``if __name__ == "__main__"`` so it is never
imported during normal test collection.
"""

from pathlib import Path

import numpy as np
from astropy.table import Table, vstack

# The parameter sets exercise a range of transit geometry: a shallow central
# transit, a deeper grazing-ish transit, and an intermediate case. Each uses the
# quadratic limb-darkening law -- the only law stellarphot uses. Orbits are
# circular (ecc=0, w=90) because that is all the fitting code supports.
REFERENCE_CASES = [
    dict(
        name="central_shallow",
        t0=0.0,
        period=5.72,
        rp=0.035,
        a=12.2,
        inc=90.0,
        u1=0.3,
        u2=0.3,
    ),
    dict(
        name="deep_grazing",
        t0=0.0,
        period=3.5,
        rp=0.12,
        a=8.0,
        inc=87.0,
        u1=0.4,
        u2=0.1,
    ),
    dict(
        name="intermediate",
        t0=0.0,
        period=10.0,
        rp=0.08,
        a=15.0,
        inc=89.0,
        u1=0.25,
        u2=0.35,
    ),
]

N_POINTS = 400

DATA_DIR = Path(__file__).parent / "data"
OUTPUT = DATA_DIR / "batman_reference_lightcurves.ecsv"


def batman_light_curve(times, case):
    import batman

    params = batman.TransitParams()
    params.t0 = case["t0"]
    params.per = case["period"]
    params.rp = case["rp"]
    params.a = case["a"]
    params.inc = case["inc"]
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"
    params.u = [case["u1"], case["u2"]]

    model = batman.TransitModel(params, times)
    return model.light_curve(params)


def build_reference_table():
    tables = []
    for case in REFERENCE_CASES:
        # A window a few transit durations wide, guaranteed to include
        # out-of-transit baseline on both sides.
        half_window = 0.12 * case["period"]
        times = np.linspace(
            case["t0"] - half_window, case["t0"] + half_window, N_POINTS
        )
        flux = batman_light_curve(times, case)
        tab = Table({"case": [case["name"]] * len(times), "time": times, "flux": flux})
        tables.append(tab)

    table = vstack(tables)
    table.meta["description"] = (
        "Reference exoplanet transit light curves generated with batman-package "
        "(quadratic limb darkening, circular orbits). Used to regression-test the "
        "pytransit replacement. Regenerate with generate_batman_reference.py."
    )
    table.meta["cases"] = REFERENCE_CASES
    return table


if __name__ == "__main__":
    import batman  # noqa: F401  fail fast with a clear error if it is missing

    DATA_DIR.mkdir(exist_ok=True)
    table = build_reference_table()
    table.write(OUTPUT, format="ascii.ecsv", overwrite=True)
    print(f"Wrote {len(table)} rows for {len(REFERENCE_CASES)} cases to {OUTPUT}")
