# stellarphot — Entry Points

stellarphot has **no command-line scripts**. There are three ways in:

1. **The Jupyter launcher** — `.jp_app_launcher_stellarphot.yaml` registers ten
   notebooks with `jupyter-app-launcher`, organized into three catalogs.
2. **The public import API** — `from stellarphot import ...` re-exports the
   core data classes and catalog fetchers.
3. **Direct library use** — importing classes such as
   `stellarphot.photometry.AperturePhotometry` or
   `stellarphot.transit_fitting.TransitModelFit` from subpackages.

**Legend** — the node types used on this page:

```mermaid
flowchart LR
    classDef nb fill:#fff8e1,stroke:#f9a825,color:#212121
    classDef widget fill:#fff3e0,stroke:#ef6c00,color:#212121
    classDef api fill:#e8f5e9,stroke:#2e7d32,color:#212121
    classDef pkg fill:#e1f5fe,stroke:#0277bd,color:#212121
    classDef file fill:#eceff1,stroke:#90a4ae,stroke-dasharray: 5 5,color:#212121

    l_nb["Launcher notebook"]:::nb
    l_widget["stellarphot widget"]:::widget
    l_api["Importable function<br/>or class (public API)"]:::api
    l_pkg["Import entry point<br/>or settings model"]:::pkg
    l_file["A file on disk"]:::file

    l_nb ~~~ l_widget ~~~ l_api ~~~ l_pkg ~~~ l_file
```

- **Solid arrows** — opens, instantiates, calls, or returns.
- **Dashed arrows** — loading from or saving to a file.

## Launcher notebooks and the widgets behind them

Each launcher entry opens a notebook whose first cells instantiate one main
widget or function; the diagram shows what each notebook puts on screen.

*Arrows: **solid** = opens / instantiates / calls / returns; **dashed** = loading from or saving to a file.*

```mermaid
flowchart LR
    classDef nb fill:#fff8e1,stroke:#f9a825,color:#212121
    classDef widget fill:#fff3e0,stroke:#ef6c00,color:#212121
    classDef api fill:#e8f5e9,stroke:#2e7d32,color:#212121

    subgraph sg_setup["catalog: Stellarphot setup"]
        direction TB
        nb01["1 - Saveable settings"]:::nb
    end

    subgraph sg_phot["catalog: Stellarphot photometry"]
        direction TB
        nb02["2 - Seeing profile and<br/>comparison stars"]:::nb
        nb03["3 - Review settings"]:::nb
        nb04["4 - Launch photometry"]:::nb
    end

    subgraph sg_tools["catalog: Stellarphot analysis and tools"]
        direction TB
        nb_target["Generate TESS target list"]:::nb
        nb_flux["Calculate relative flux"]:::nb
        nb_calib["Calibrate magnitudes"]:::nb
        nb_fit1["Initial exoplanet model"]:::nb
        nb_fit2["Second exoplanet model fit"]:::nb
        nb_exotic["Exoplanet model fit<br/>with EXOTIC"]:::nb
    end

    review1["ReviewSettings([Camera,<br/>Observatory, PassbandMap])"]:::widget
    cas["ComparisonAndSeeing"]:::widget
    review3["ReviewSettings(7 settings models)"]:::widget
    runner["PhotometryRunner"]:::widget
    tps["TessPhotometrySetup"]:::widget
    addflux["add_relative_flux_column()"]:::api
    calib["transform_to_catalog() and<br/>calculate_transform_coefficients()"]:::api
    fitstack["TessAnalysisInputControls +<br/>TransitModelFit + TransitModelOptions"]:::widget
    exotic["exotic_settings_widget() +<br/>populate_TOI_boxes()"]:::widget

    apphot["AperturePhotometry"]:::api

    nb01 --> review1
    nb02 --> cas
    nb03 --> review3
    nb04 --> runner
    runner -->|"papermill runs<br/>photometry_runner.ipynb"| apphot
    nb_target --> tps
    nb_flux --> addflux
    nb_calib --> calib
    nb_fit1 --> fitstack
    nb_fit2 --> fitstack
    nb_exotic --> exotic

    click nb01 href "../stellarphot/notebooks/01-initial-settings.ipynb" "01-initial-settings.ipynb"
    click nb02 href "../stellarphot/notebooks/02-seeing-and-comparison.ipynb" "02-seeing-and-comparison.ipynb"
    click nb03 href "../stellarphot/notebooks/03-final-review-of-settings.ipynb" "03-final-review-of-settings.ipynb"
    click nb04 href "../stellarphot/notebooks/04-launch-photometry.ipynb" "04-launch-photometry.ipynb"
    click nb_target href "../stellarphot/notebooks/tess-target-source-list-generator.ipynb" "tess-target-source-list-generator.ipynb"
    click nb_flux href "../stellarphot/notebooks/relative-flux-calculation.ipynb" "relative-flux-calculation.ipynb"
    click nb_calib href "../stellarphot/notebooks/transform-to-appas-dr9.ipynb" "transform-to-appas-dr9.ipynb"
    click nb_fit1 href "../stellarphot/notebooks/tess-initial-model-fit.ipynb" "tess-initial-model-fit.ipynb"
    click nb_fit2 href "../stellarphot/notebooks/tess-second-model-fit.ipynb" "tess-second-model-fit.ipynb"
    click nb_exotic href "../stellarphot/notebooks/tess-EXOTIC-fit.ipynb" "tess-EXOTIC-fit.ipynb"
    click review1 href "../stellarphot/settings/custom_widgets.py" "custom_widgets.py"
    click cas href "../stellarphot/gui_tools/profile_and_comps.py" "profile_and_comps.py"
    click review3 href "../stellarphot/settings/custom_widgets.py" "custom_widgets.py"
    click runner href "../stellarphot/settings/custom_widgets.py" "custom_widgets.py"
    click tps href "../stellarphot/settings/custom_widgets.py" "custom_widgets.py"
    click addflux href "../stellarphot/differential_photometry/aij_rel_fluxes.py" "aij_rel_fluxes.py"
    click calib href "../stellarphot/utils/magnitude_transforms.py" "magnitude_transforms.py"
    click fitstack href "../stellarphot/transit_fitting/core.py" "transit_fitting/core.py"
    click exotic href "../stellarphot/transit_fitting/gui.py" "transit_fitting/gui.py"
    click apphot href "../stellarphot/photometry/photometry.py" "photometry.py"
```

What sits behind each widget:

| Launcher entry | Notebook | Main objects used |
|---|---|---|
| 1 - Saveable settings | [`01-initial-settings.ipynb`](../stellarphot/notebooks/01-initial-settings.ipynb) | `ReviewSettings([Camera, Observatory, PassbandMap])` |
| 2 - Seeing profile and comparison stars | [`02-seeing-and-comparison.ipynb`](../stellarphot/notebooks/02-seeing-and-comparison.ipynb) | `ComparisonAndSeeing` (= `SeeingProfileWidget` + `ComparisonViewer`) |
| 3 - Review settings | [`03-final-review-of-settings.ipynb`](../stellarphot/notebooks/03-final-review-of-settings.ipynb) | `ReviewSettings` over all seven `PhotometrySettings` component models |
| 4 - Launch photometry | [`04-launch-photometry.ipynb`](../stellarphot/notebooks/04-launch-photometry.ipynb) | `PhotometryRunner` → papermill → `AperturePhotometry` |
| Generate TESS target list | [`tess-target-source-list-generator.ipynb`](../stellarphot/notebooks/tess-target-source-list-generator.ipynb) | `TessPhotometrySetup` → `tess_photometry_setup()` |
| Calculate relative flux | [`relative-flux-calculation.ipynb`](../stellarphot/notebooks/relative-flux-calculation.ipynb) | `add_relative_flux_column()`, `FitsOpener`, `Spinner` |
| Calibrate magnitudes | [`transform-to-appas-dr9.ipynb`](../stellarphot/notebooks/transform-to-appas-dr9.ipynb) | `transform_to_catalog()`, `vsx_vizier()`, `PhotometryData` |
| Initial / Second exoplanet model | [`tess-initial-model-fit.ipynb`](../stellarphot/notebooks/tess-initial-model-fit.ipynb), [`tess-second-model-fit.ipynb`](../stellarphot/notebooks/tess-second-model-fit.ipynb) | `TessAnalysisInputControls`, `filter_by_dates()`, `TransitModelFit`, `TransitModelOptions`, `TOI`, `plot_transit_lightcurve()` |
| Exoplanet model fit with EXOTIC | [`tess-EXOTIC-fit.ipynb`](../stellarphot/notebooks/tess-EXOTIC-fit.ipynb) | `exotic_settings_widget()`, `populate_TOI_boxes()`, `get_values_from_widget()`, `generate_json_file_name()` |

## Public import API

*Arrows: **solid** = opens / instantiates / calls / returns; **dashed** = loading from or saving to a file.*

```mermaid
flowchart LR
    classDef pkg fill:#e1f5fe,stroke:#0277bd,color:#212121
    classDef api fill:#e8f5e9,stroke:#2e7d32,color:#212121
    classDef widget fill:#fff3e0,stroke:#ef6c00,color:#212121

    top["from stellarphot import ..."]:::pkg

    subgraph sg_tables["data classes (core.py)"]
        direction TB
        bet["BaseEnhancedTable"]:::api
        pdata["PhotometryData"]:::api
        cdata["CatalogData"]:::api
        sldata["SourceListData"]:::api
    end

    subgraph sg_cats["catalog fetchers (core.py)"]
        direction TB
        apass["apass_dr9()"]:::api
        refcat["refcat2()"]:::api
        vsx["vsx_vizier()"]:::api
    end

    top --> bet
    top --> pdata
    top --> cdata
    top --> sldata
    top --> apass
    top --> refcat
    top --> vsx

    subgraph sg_deep["main subpackage imports"]
        direction TB
        apphot["stellarphot.photometry<br/>AperturePhotometry"]:::api
        tmf["stellarphot.transit_fitting<br/>TransitModelFit"]:::api
        relflux["stellarphot.differential_photometry<br/>calc_aij_relative_flux()"]:::api
        cview["stellarphot.gui_tools<br/>ComparisonViewer, SeeingProfileWidget"]:::widget
        rsw["stellarphot.settings.custom_widgets<br/>ReviewSettings, PhotometryRunner"]:::widget
        waavso["stellarphot.io<br/>write_aavso_extended(), TOI"]:::api
    end

    apphot -->|"returns"| pdata
    relflux -->|"adds column to"| pdata
    cview -->|"writes"| sldata

    click top href "../stellarphot/__init__.py" "stellarphot/__init__.py"
    click bet href "../stellarphot/core.py" "core.py"
    click pdata href "../stellarphot/core.py" "core.py"
    click cdata href "../stellarphot/core.py" "core.py"
    click sldata href "../stellarphot/core.py" "core.py"
    click apass href "../stellarphot/core.py" "core.py"
    click refcat href "../stellarphot/core.py" "core.py"
    click vsx href "../stellarphot/core.py" "core.py"
    click apphot href "../stellarphot/photometry/photometry.py" "photometry.py"
    click tmf href "../stellarphot/transit_fitting/core.py" "transit_fitting/core.py"
    click relflux href "../stellarphot/differential_photometry/aij_rel_fluxes.py" "aij_rel_fluxes.py"
    click cview href "../stellarphot/gui_tools/" "gui_tools/"
    click rsw href "../stellarphot/settings/custom_widgets.py" "custom_widgets.py"
    click waavso href "../stellarphot/io/" "io/"
```

*Source: [__init__.py](../stellarphot/__init__.py) (the re-exports), [core.py](../stellarphot/core.py) (the data classes and catalog fetchers).*

## The main programmatic entry point: `AperturePhotometry`

`AperturePhotometry` is configured by a single `PhotometrySettings` object,
usually loaded from `photometry_settings.json` via
`PhotometryWorkingDirSettings`.

*Arrows: **solid** = opens / instantiates / calls / returns; **dashed** = loading from or saving to a file.*

```mermaid
flowchart LR
    classDef cls fill:#e1f5fe,stroke:#0277bd,color:#212121
    classDef api fill:#e8f5e9,stroke:#2e7d32,color:#212121
    classDef file fill:#eceff1,stroke:#90a4ae,stroke-dasharray: 5 5,color:#212121

    json["photometry_settings.json"]:::file
    pwds["PhotometryWorkingDirSettings<br/>.load()"]:::cls
    ps["PhotometrySettings"]:::cls

    subgraph sg_parts["component models"]
        direction TB
        cam["camera: Camera"]:::cls
        obs["observatory: Observatory"]:::cls
        apert["photometry_apertures:<br/>PhotometryApertures"]:::cls
        srcloc["source_location_settings:<br/>SourceLocationSettings"]:::cls
        opt["photometry_optional_settings:<br/>PhotometryOptionalSettings"]:::cls
        pbm["passband_map:<br/>PassbandMap or None"]:::cls
        logset["logging_settings:<br/>LoggingSettings"]:::cls
    end

    apinst["AperturePhotometry(settings=...)"]:::api
    apcall["__call__(file_or_directory,<br/>reject_unmatched, object_of_interest)"]:::api
    result["PhotometryData"]:::api

    json -.-> pwds
    pwds --> ps
    ps --> cam
    ps --> obs
    ps --> apert
    ps --> srcloc
    ps --> opt
    ps --> pbm
    ps --> logset
    ps --> apinst
    apinst --> apcall
    apcall --> result

    click pwds href "../stellarphot/settings/settings_files.py" "settings_files.py"
    click ps href "../stellarphot/settings/models.py" "models.py"
    click cam href "../stellarphot/settings/models.py" "models.py"
    click obs href "../stellarphot/settings/models.py" "models.py"
    click apert href "../stellarphot/settings/models.py" "models.py"
    click srcloc href "../stellarphot/settings/models.py" "models.py"
    click opt href "../stellarphot/settings/models.py" "models.py"
    click pbm href "../stellarphot/settings/models.py" "models.py"
    click logset href "../stellarphot/settings/models.py" "models.py"
    click apinst href "../stellarphot/photometry/photometry.py" "photometry.py"
    click apcall href "../stellarphot/photometry/photometry.py" "photometry.py"
    click result href "../stellarphot/core.py" "core.py"
```

*Source: [photometry.py](../stellarphot/photometry/photometry.py), [models.py](../stellarphot/settings/models.py), [settings_files.py](../stellarphot/settings/settings_files.py), [core.py](../stellarphot/core.py).*

### `AperturePhotometry.__call__` arguments

| Argument | Type / default | Meaning |
|---|---|---|
| `file_or_directory` | `str \| Path` (required) | A single FITS file → `single_image_photometry()`; a directory → `multi_image_photometry()` over every matching image |
| `logline` | `str`, `"single_image_photometry:"` | Prefix for log messages (single-image only) |
| `reject_unmatched` | `bool`, `True` | Drop sources not detected on every image (multi-image only) |
| `object_of_interest` | `str`, `None` | Only process files whose `OBJECT` header matches (multi-image only) |

### Settings model fields (what the JSON file / widgets configure)

| Model | Fields |
|---|---|
| `Camera` | `name`, `data_unit`, `gain`, `read_noise`, `dark_current`, `pixel_scale`, `max_data_value` |
| `Observatory` | `name`, `latitude`, `longitude`, `elevation`, `AAVSO_code`, `TESS_telescope_code` |
| `PhotometryApertures` | `variable_aperture`, `radius`, `gap`, `annulus_width`, `fwhm_estimate` |
| `SourceLocationSettings` | `source_list_file`, `use_coordinates` (`"sky"` or `"pixel"`), `shift_tolerance` |
| `PhotometryOptionalSettings` | `include_dig_noise`, `reject_too_close`, `reject_background_outliers`, `fwhm_method` (`fit`/`profile`/`moments`), `partial_pixel_method` (`exact`/`center`/`subpixel`) |
| `PassbandMap` | `name`, `your_filter_names_to_aavso` (list of `PassbandMapEntry`: instrument filter → AAVSO filter) |
| `LoggingSettings` | `logfile`, `console_log` |
| `PhotometryRunSettings` | `directory_with_images`, `photometry_settings_file`, `reject_unmatched`, `object_of_interest` (parameters papermill passes to `photometry_runner.ipynb`) |

## Other notable callable entry points

| Entry point | Module | Key arguments | What's behind it |
|---|---|---|---|
| `source_detection()` | `stellarphot.photometry` | `ccd`, `fwhm`, `sigma`, `iters`, `threshold`, `find_fwhm` | `DAOStarFinder` + `compute_fwhm()` → returns `SourceListData` |
| `calc_aij_relative_flux()` | `stellarphot.differential_photometry` | `star_data`, `comp_stars`, `in_place`, `coord_column`, `star_id_column` | AIJ-style comparison-star ensemble flux |
| `TransitModelFit.setup_model()` / `.fit()` | `stellarphot.transit_fitting` | `t0`, `depth`, `duration`, `period`, `inclination`, plus airmass/width/sky detrending | batman transit model + `VariableArgsFitter` (scipy leastsq) |
| `transform_to_catalog()` | `stellarphot.utils` | photometry table, passband options, fit order | Cross-match to APASS DR9 / RefCat2 and fit `calibrated_from_instrumental()` |
| `write_aavso_extended()` | `stellarphot.io` | photometry table, destination path, header info | Formats and writes an AAVSO extended-format submission file |
| `tess_photometry_setup()` | `stellarphot.io` | `tic_id` or `TOI_object`, `overwrite` | Queries MAST/ExoFOP, writes `TIC-<id>-info.json` and `TIC-<id>-source-list-input.ecsv` |
| `apass_dr9()`, `refcat2()`, `vsx_vizier()` | `stellarphot` | a WCS or `SkyCoord` (+ search radius) | Vizier/XMatch query → `CatalogData` |
