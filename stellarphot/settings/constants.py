# Though these constants are used in the tests they are needed in a few
# non-test places too. Putting them here ensures they can be imported
# without needing test dependencies.

TEST_APERTURE_SETTINGS = dict(
    variable_aperture=False,
    radius=5,
    gap=10,
    annulus_width=15,
    fwhm_estimate=3.2,
)

TEST_CAMERA_VALUES = dict(
    data_unit="adu",
    gain="2.0 electron / adu",
    name="test camera",
    read_noise="10.0 electron",
    dark_current="0.01 electron / s",
    pixel_scale="0.563 arcsec / pix",
    max_data_value="50000.0 adu",
)

TEST_EXOPLANET_SETTINGS = dict(
    epoch={
        "jd1": 0.0,
        "jd2": 0.0,
        "format": "jd",
        "scale": "utc",
        "precision": 3,
        "in_subfmt": "*",
        "out_subfmt": "*",
    },
    period="0.0 min",
    identifier="a planet",
    coordinate={
        "ra": "0d00m00s",
        "dec": "0d00m00s",
        "representation_type": "spherical",
        "frame": "icrs",
    },
    depth=0,
    duration="0.0 min",
)

TEST_OBSERVATORY_SETTINGS = dict(
    name="test observatory",
    longitude="43d00m00s",
    latitude="45d00m00s",
    elevation="311.0 m",
    AAVSO_code="test",
    TESS_telescope_code="tess test",
)

# The first setting here is required, the rest are optional. The optional
# settings below are different than the defaults in the model definition.
TEST_PHOTOMETRY_OPTIONS = dict(
    include_dig_noise=False,
    reject_too_close=False,
    reject_background_outliers=False,
    fwhm_method="fit",
    partial_pixel_method="center",
)

TEST_PASSBAND_MAP = dict(
    name="Example map",
    your_filter_names_to_aavso=[
        dict(
            your_filter_name="V",
            aavso_filter_name="V",
        ),
        dict(
            your_filter_name="B",
            aavso_filter_name="B",
        ),
        dict(
            your_filter_name="rp",
            aavso_filter_name="SR",
        ),
    ],
)

TEST_LOGGING_SETTINGS = dict(
    logfile="test.log",
    console_log=False,
)

TEST_SOURCE_LOCATION_SETTINGS = dict(
    shift_tolerance=5,
    source_list_file="test.ecsv",
    use_coordinates="pixel",
)

TEST_PHOTOMETRY_SETTINGS = dict(
    camera=TEST_CAMERA_VALUES,
    observatory=TEST_OBSERVATORY_SETTINGS,
    photometry_apertures=TEST_APERTURE_SETTINGS,
    source_location_settings=TEST_SOURCE_LOCATION_SETTINGS,
    photometry_optional_settings=TEST_PHOTOMETRY_OPTIONS,
    passband_map=TEST_PASSBAND_MAP,
    logging_settings=TEST_LOGGING_SETTINGS,
)
