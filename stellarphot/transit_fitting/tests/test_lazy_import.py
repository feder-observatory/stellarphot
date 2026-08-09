import subprocess
import sys


def test_tess_import_does_not_import_pytransit():
    # The GUI import chain reaches stellarphot.transit_fitting.io on the way
    # to get_tic_info. Importing that package must not drag in pytransit,
    # which takes seconds to import and emits warnings that would show up in
    # notebook cells that only use the GUI. A subprocess is used because
    # pytransit may already be in sys.modules of the test process.
    code = "import sys; import stellarphot.io.tess; print('pytransit' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_transit_model_symbols_importable_from_package():
    from stellarphot import transit_fitting

    assert transit_fitting.TransitModelFit is not None
    assert transit_fitting.TransitModelOptions is not None
    # dir() must include the lazy names, e.g. for the docs build.
    assert "TransitModelFit" in dir(transit_fitting)
    assert "TransitModelOptions" in dir(transit_fitting)
