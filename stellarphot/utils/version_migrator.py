import numpy as np
from astropy import units as u
from astropy.time import Time
from packaging.version import Version

from stellarphot import PhotometryData
from stellarphot.settings import Camera, Observatory, PassbandMap


class VersionMigrator:
    """
    Migrate data from one stellarphot major version to another.

    Parameters
    ----------
    from_version : str
        The version of the data to migrate from.

    to_version : str
        The version of the data to migrate to.
    """

    known_versions = ("1", "2")

    def __init__(
        self,
        from_version: str = "1",
        to_version: str = "2",
        camera: Camera = None,
        observatory: Observatory = None,
        passband_map: PassbandMap = None,
    ):
        self._from_version = Version(from_version)
        self._to_version = Version(to_version)
        if str(self.from_version.major) not in self.known_versions:
            raise ValueError(
                f"Unknown version: {self.from_version=}. "
                f"Valid versions are: {self.known_versions}"
            )

        if str(self.to_version.major) not in self.known_versions:
            raise ValueError(
                f"Unknown version: {self.to_version=}. "
                f"Valid versions are: {self.known_versions}"
            )

        if self.from_version >= self.to_version:
            raise ValueError(
                "Can only migrate from a lower version to a higher version"
            )

        self.camera = camera
        self.observatory = observatory
        self.passband_map = passband_map

    @property
    def from_version(self):
        return self._from_version

    @property
    def to_version(self):
        return self._to_version

    def migrate(self, data):
        """
        Migrate data from one version to another.
        """
        if self.from_version.major == 1 and self.to_version.major == 2:
            return self._migrate_v1_v2(data)

    def _migrate_v1_v2(self, data):
        """
        Migrate data from version 1 to version 2.
        """
        new_data = data.copy()

        unitify = {
            "fwhm_x": u.pixel,
            "fwhm_y": u.pixel,
            "width": u.pixel,
            "aperture": u.pixel,
            "annulus_inner": u.pixel,
            "annulus_outer": u.pixel,
            "aperture_area": u.pixel,
            "annulus_area": u.pixel,
            "noise-aij": u.electron,
            "annulus_sum": u.adu,
            "aperture_sum": u.adu,
            "aperture_net_flux": u.adu,
            "noise": u.adu,
            "sky_per_pix_avg": u.adu / u.pixel,
            "sky_per_pix_med": u.adu / u.pixel,
            "sky_per_pix_std": u.adu / u.pixel,
            "RA": u.deg,
            "Dec": u.deg,
            "xcenter": u.pixel,
            "ycenter": u.pixel,
            "exposure": u.second,
        }
        for un in unitify:
            # Note that this deliberately discards any unit that might have been
            # present in the original data. This is a stellarphot-specific migration
            # tool, not something more general, and the choices above are the correct
            # units for the data.
            new_data[un] = new_data[un].value * unitify[un]

        new_data["date-obs"] = Time(new_data["date-obs"], scale="utc")

        v1_to_v2_col_renames = {
            "aperture_net_flux": "aperture_net_cnts",
            "BJD": "bjd",
            "noise": "noise_cnts",
            "noise-aij": "noise_electrons",
            "filter": "passband",
            "RA": "ra",
            "Dec": "dec",
        }

        # THe new night calculation is better, so drop the old one
        del new_data["night"]

        # Handle changes in instrumental magnitude column names
        mag_columns = [col for col in new_data.colnames if col.startswith("mag_inst")]
        mag_cols_have_filter_names = any(
            mag.split("_")[-1] in new_data["filter"] for mag in mag_columns
        )

        if mag_cols_have_filter_names:
            new_mag_col_data = np.zeros_like(new_data["date-obs"])
            for col in mag_columns:
                passband = col.split("_")[-1]
                this_passband = new_data["filter"] == passband
                new_mag_col_data[this_passband] = new_data[col][this_passband]
            new_data["mag_inst"] = new_mag_col_data
            new_data.remove_columns(mag_columns)

        return PhotometryData(
            input_data=new_data,
            camera=self.camera,
            observatory=self.observatory,
            passband_map=self.passband_map,
            colname_map=v1_to_v2_col_renames,
        )
