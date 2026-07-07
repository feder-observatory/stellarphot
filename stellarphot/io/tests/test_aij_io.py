import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from stellarphot.io.aij import ApertureAIJ, ApertureFileAIJ, generate_aij_table


def test_aperture_eq():
    ap1 = ApertureAIJ()
    ap2 = ApertureAIJ()
    ap3 = ApertureAIJ()
    ap3.rback1 = 27.0
    assert ap1 == ap2
    assert ap2 == ap3


def test_aperture_file_content():
    # Test that the format of the generated aperture file
    # matches the expected format
    ap = ApertureFileAIJ()
    ref_data = get_pkg_data_filename("data/apertures_as_table.csv")
    ref_table = Table.read(ref_data)

    ap.multiaperture.xapertures = ref_table["x"]

    # AIJ has origin in different place than the reference table.
    ap.multiaperture.yapertures = np.around((4096 - ref_table["y"]), decimals=4)

    ap.multiaperture.raapertures = ref_table["ra"]
    ap.multiaperture.decapertures = ref_table["dec"]

    ap.multiaperture.isrefstar = ref_table["isrefstar"]
    ap.multiaperture.centroidstar = ref_table["centroidstar"]
    ap.multiaperture.isalignstar = ref_table["isalignstar"]

    ap.multiaperture.absmagapertures = ref_table["absmag"]

    ref_aperture_file = get_pkg_data_filename("data/aij-sample-apertures.aperture")

    ref_apertures = ApertureFileAIJ.read(ref_aperture_file)

    assert ref_apertures == ap


def test_aperture_creation_from_table():
    # Check that generating an aperture object from an
    # aperture table gives the right result.

    ref_data = get_pkg_data_filename("data/apertures_as_table.csv")
    ref_table = Table.read(ref_data)

    # Need to create a coord column
    coordinates = SkyCoord(ra=ref_table["ra"], dec=ref_table["dec"], unit="degree")
    ref_table["coord"] = coordinates

    # Delete some columns that are not in the usual aperture table
    del (
        ref_table["ra"],
        ref_table["dec"],
        ref_table["isalignstar"],
        ref_table["centroidstar"],
    )

    # Generate marker names that match those in aperture file
    # Note that the csv reads in the True/False column as text,
    # not as bool.
    ref_table["marker name"] = [
        "APASS comparison" if v == "True" else "TESS Targets"
        for v in ref_table["isrefstar"]
    ]

    del ref_table["isrefstar"]

    ap_info = ApertureAIJ()
    ap_aij = ApertureFileAIJ.from_table(
        ref_table,
        aperture_rad=ap_info.radius,
        inner_annulus=ap_info.rback1,
        outer_annulus=ap_info.rback2,
    )
    ref_aperture_file = get_pkg_data_filename("data/aij-sample-apertures.aperture")

    ref_apertures = ApertureFileAIJ.read(ref_aperture_file)

    assert ref_apertures == ap_aij


def _make_photometry_table(star_ids, ras, decs):
    # Build a minimal photometry table with the columns that
    # generate_aij_table expects, with one row per star.
    n_stars = len(star_ids)
    phot_table = Table(
        {
            "star_id": star_ids,
            "RA": ras,
            "Dec": decs,
            "date-obs": ["2024-01-01T01:02:03.456"] * n_stars,
            "airmass": [1.2] * n_stars,
            "BJD": [2460310.54] * n_stars,
            "exposure": [30.0] * n_stars,
            "filter": ["ip"] * n_stars,
            "aperture": [10.0] * n_stars,
            "annulus_inner": [15.0] * n_stars,
            "annulus_outer": [25.0] * n_stars,
            "xcenter": [100.0] * n_stars,
            "ycenter": [200.0] * n_stars,
            "aperture_net_counts": [10000.0] * n_stars,
            "aperture_area": [314.0] * n_stars,
            "noise-aij": [100.0] * n_stars,
            "snr": [100.0] * n_stars,
            "sky_per_pix_avg": [10.0] * n_stars,
            "annulus_area": [1256.0] * n_stars,
            "fwhm_x": [5.0] * n_stars,
            "fwhm_y": [5.0] * n_stars,
            "width": [5.0] * n_stars,
            "relative_flux": [0.5] * n_stars,
            "relative_flux_error": [0.01] * n_stars,
            "relative_flux_snr": [50.0] * n_stars,
            "comparison counts": [20000.0] * n_stars,
            "comparison error": [140.0] * n_stars,
        }
    )
    return phot_table


def _make_comparison_table(ras, decs, marker_names):
    comp_table = Table(
        {
            "coord": SkyCoord(ra=ras, dec=decs, unit="degree"),
            "marker name": marker_names,
        }
    )
    return comp_table


def test_generate_aij_table_mixed_target_and_comps():
    # A comparison table that contains both the target and the
    # comparison stars, each with the appropriate marker name, should
    # lead to the target getting "_T" columns and the comparison stars
    # getting "_C" columns.
    target_ra, target_dec = 50.0, 45.0
    comp_ras = [10.0, 10.02]
    comp_decs = [45.0, 45.01]

    phot_table = _make_photometry_table(
        star_ids=[1, 2, 3],
        ras=[target_ra] + comp_ras,
        decs=[target_dec] + comp_decs,
    )
    comparison_table = _make_comparison_table(
        ras=[target_ra] + comp_ras,
        decs=[target_dec] + comp_decs,
        marker_names=["TESS Targets", "APASS comparison", "APASS comparison"],
    )

    aij_table = generate_aij_table(phot_table, comparison_table)

    assert "rel_flux_T1" in aij_table.colnames
    assert "rel_flux_C2" in aij_table.colnames
    assert "rel_flux_C3" in aij_table.colnames


def test_generate_aij_table_target_not_in_comparison_table():
    # Regression test for #595 -- when the comparison table contains
    # only comparison stars, a target far from every comparison star
    # was matched to its nearest comparison star, however far away,
    # and mislabeled as a comparison star. The result was an AIJ table
    # with no "_T" columns at all.
    target_ra, target_dec = 50.0, 45.0
    comp_ras = [10.0, 10.02]
    comp_decs = [45.0, 45.01]

    phot_table = _make_photometry_table(
        star_ids=[1, 2, 3],
        ras=[target_ra] + comp_ras,
        decs=[target_dec] + comp_decs,
    )
    # The comparison table contains only the comparison stars, tens of
    # degrees away from the target.
    comparison_table = _make_comparison_table(
        ras=comp_ras,
        decs=comp_decs,
        marker_names=["APASS comparison", "APASS comparison"],
    )

    aij_table = generate_aij_table(phot_table, comparison_table)

    # The target must be labeled as a target, not a comparison star...
    assert "rel_flux_T1" in aij_table.colnames
    assert "rel_flux_C1" not in aij_table.colnames

    # ...and the comparison stars should still be labeled as comparisons.
    assert "rel_flux_C2" in aij_table.colnames
    assert "rel_flux_C3" in aij_table.colnames
