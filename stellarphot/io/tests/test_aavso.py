import pytest

from stellarphot.io import AAVSOExtendedFileFormat

DEFAULT_OBSCODE = "ABCDE"


def test_no_obscode_raises_error():
    with pytest.raises(TypeError, match="observer_code"):
        aef = AAVSOExtendedFileFormat()


def test_default_values():
    aef = AAVSOExtendedFileFormat(DEFAULT_OBSCODE)
    assert aef.delim == ","
    assert len(aef.data) == 0


def test_setting_type_raises_error():
    aef = AAVSOExtendedFileFormat(DEFAULT_OBSCODE)
    with pytest.raises(AttributeError, match="can't set attribute"):
        aef.type = 'STD'