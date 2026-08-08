from copy import deepcopy

import pytest
from astropy.io.misc.yaml import AstropyDumper

from stellarphot.settings import models
from stellarphot.settings.models import (
    PHOTOMETRY_SETTINGS_FORMAT_VERSION,
    FwhmMethods,
    NewerFormatError,
    PhotometrySettingsWarning,
)
from stellarphot.table_representations import (
    _generate_old_table_representers,
    deserialize_models_in_table_meta,
    serialize_models_in_table_meta,
)


class TestNonModelEntriesInAll:
    """
    ``models.__all__`` lists some classes that are not pydantic models --
    exceptions, warning categories, and the FwhmMethods enum. The
    table-metadata machinery must skip those rather than crash on them.
    """

    @pytest.mark.parametrize("name", ["NewerFormatError", "FwhmMethods"])
    def test_deserialize_leaves_non_model_entries_alone(self, name):
        # A metadata entry naming a non-model class is not something we can
        # deserialize; it must pass through untouched.
        meta = {"x": {"_model_name": name, "value": "{}"}}
        expected = deepcopy(meta)
        deserialize_models_in_table_meta(meta)
        assert meta == expected

    def test_serialize_leaves_non_model_instances_alone(self):
        # FwhmMethods is a StrEnum, so an instance of it in the metadata is
        # a plain value, not a model to be serialized.
        meta = {"method": FwhmMethods.FIT}
        serialize_models_in_table_meta(meta)
        assert meta["method"] == FwhmMethods.FIT

    def test_old_style_representers_register_only_models(self):
        # The YAML constructors for old-style tables call model_validate_json
        # on whatever was registered, so registering a non-model would just
        # defer the crash to read time.
        _generate_old_table_representers()
        assert NewerFormatError not in AstropyDumper.yaml_representers
        assert FwhmMethods not in AstropyDumper.yaml_representers
        assert models.Camera in AstropyDumper.yaml_representers


class TestNewerFormatInTableMeta:
    def test_deserialize_leaves_too_new_settings_as_dict(self):
        # A table whose embedded settings were written by a newer stellarphot
        # must still be readable -- the settings are left as a plain dict
        # (exactly what astropy.table.Table.read would give) instead of
        # failing the whole read.
        too_new = {
            "_model_name": "PhotometrySettings",
            "settings_version": PHOTOMETRY_SETTINGS_FORMAT_VERSION + 1,
        }
        meta = {"photometry_settings": deepcopy(too_new)}
        with pytest.warns(PhotometrySettingsWarning, match="newer version"):
            deserialize_models_in_table_meta(meta)
        # The entry is untouched, _model_name included, so re-serializing
        # the table keeps it readable by the newer version that wrote it.
        assert meta["photometry_settings"] == too_new
