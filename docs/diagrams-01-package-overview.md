# stellarphot — Package Overview

This page maps every module in the `stellarphot` package and the import
relationships between them. The package has no command-line scripts; users
interact with it through Jupyter notebooks (launched via
`jupyter-app-launcher`), through ipywidgets-based GUI tools, or by importing
the library directly.

The modules group into these logical components:

| Component | Modules | Role |
|---|---|---|
| **Core data layer** | `core.py`, `catalogs.py`, `table_representations.py` | Validated astropy `QTable` subclasses (`PhotometryData`, `CatalogData`, `SourceListData`) in `core.py`; catalog-fetcher functions (`apass_dr9`, `vsx_vizier`, `refcat2`) in `catalogs.py`; YAML (de)serialization of settings stored in table metadata |
| **Settings & configuration** | `settings/` | Pydantic models for all configuration and saved-settings file management — pure Pydantic, no GUI imports |
| **Photometry engine** | `photometry/` | Source detection, FWHM measurement, aperture photometry pipeline |
| **Differential photometry** | `differential_photometry/` | Relative flux (AIJ-style) and variable-star magnitude calculations |
| **Transit fitting** | `transit_fitting/` | Exoplanet transit modeling (pytransit), TIC/MAST queries (the EXOTIC helper GUI moved to `gui/`) |
| **Input/output** | `io/` | AstroImageJ, AAVSO extended format, TESS/TFOP files |
| **GUI layer** | `gui/` | All notebook/widget UI, consolidated here so the rest of the package stays headless: the settings-form generator (`views.py`), settings widgets (`custom_widgets.py`), file chooser (`fits_opener.py`), seeing-profile/comparison-star widgets, and the EXOTIC helper. Requires the optional `[gui]` extra |
| **Plotting** | `plotting/` | Seeing, transit, and multi-night light-curve plots |
| **Utilities** | `utils/` | Magnitude calibration/transforms, comparison-star helpers, version migration |
| **Notebooks** | `notebooks/` | Shipped workflow notebooks (the user-facing entry points) |

**Legend** — the node types used on this page:

```mermaid
flowchart LR
    classDef data fill:#e1f5fe,stroke:#0277bd,color:#212121
    classDef engine fill:#e8f5e9,stroke:#2e7d32,color:#212121
    classDef ui fill:#fff3e0,stroke:#ef6c00,color:#212121
    classDef output fill:#f3e5f5,stroke:#7b1fa2,color:#212121
    classDef nbstyle fill:#fff8e1,stroke:#f9a825,color:#212121
    classDef external fill:#eceff1,stroke:#90a4ae,stroke-dasharray: 5 5,color:#212121

    l_data["Core data layer"]:::data
    l_engine["Processing engine"]:::engine
    l_ui["UI layer"]:::ui
    l_output["Output<br/>(io, plotting)"]:::output
    l_nb["Launcher notebook"]:::nbstyle
    l_ext["Outside the<br/>cluster shown"]:::external

    l_data ~~~ l_engine ~~~ l_ui ~~~ l_output ~~~ l_nb ~~~ l_ext
```

- **Solid arrows** — an import, pointing from the importing module to the one
  it imports.
- **Dashed arrows** — notebooks driving a UI layer, or other looser links.
- Files are grouped into labeled **subgraphs** by subpackage in the
  cluster diagrams.

## Subpackage dependency map

Arrows point from the importing module to the module it imports.
The notebooks drive the GUI layers (dashed arrows) and, through them, the rest
of the package.

*Arrows: **solid** = an import (importer → imported); **dashed** = a notebook driving a UI layer, or a looser link.*

```mermaid
flowchart LR
    classDef data fill:#e1f5fe,stroke:#0277bd,color:#212121
    classDef engine fill:#e8f5e9,stroke:#2e7d32,color:#212121
    classDef ui fill:#fff3e0,stroke:#ef6c00,color:#212121
    classDef output fill:#f3e5f5,stroke:#7b1fa2,color:#212121

    nb["notebooks/<br/>(10 launcher notebooks)"]:::ui
    gui["gui/"]:::ui
    sett["settings/"]:::ui
    phot["photometry/"]:::engine
    diff["differential_photometry/"]:::engine
    tfit["transit_fitting/"]:::engine
    uts["utils/"]:::engine
    plt["plotting/"]:::output
    iop["io/"]:::output
    corem["core.py"]:::data
    catm["catalogs.py"]:::data
    trep["table_representations.py"]:::data

    nb -.->|"drives"| gui
    nb -.->|"drives"| sett

    gui --> sett
    gui --> phot
    gui --> plt
    gui --> iop
    gui --> uts
    gui --> corem
    gui -->|"transit_fitting_gui uses get_tic_info"| tfit

    phot --> corem
    phot --> sett
    diff --> corem
    uts --> corem
    uts -->|"apass_dr9, vsx_vizier, refcat2"| catm
    uts --> sett
    plt --> sett
    iop -.->|"core (lazy, io.tess)"| corem
    iop --> sett
    iop -->|"get_tic_info"| tfit

    corem -->|"io.aavso"| iop
    corem --> sett
    corem --> trep
    catm -->|"CatalogData"| corem
    catm -->|"PassbandMap"| sett
    trep --> sett

    click nb href "../stellarphot/notebooks/" "notebooks/"
    click gui href "../stellarphot/gui/" "gui/"
    click sett href "../stellarphot/settings/" "settings/"
    click phot href "../stellarphot/photometry/" "photometry/"
    click diff href "../stellarphot/differential_photometry/" "differential_photometry/"
    click tfit href "../stellarphot/transit_fitting/" "transit_fitting/"
    click uts href "../stellarphot/utils/" "utils/"
    click plt href "../stellarphot/plotting/" "plotting/"
    click iop href "../stellarphot/io/" "io/"
    click corem href "../stellarphot/core.py" "core.py"
    click catm href "../stellarphot/catalogs.py" "catalogs.py"
    click trep href "../stellarphot/table_representations.py" "table_representations.py"
```

*Source: the [stellarphot/](../stellarphot/) package — each box is a subpackage directory or top-level module.*

Notes:

- The notebooks also import `differential_photometry`, `transit_fitting`,
  `io`, `plotting`, `utils`, and the top-level `stellarphot` package directly;
  only the two main dashed edges are drawn to keep the diagram readable.
- `core.py` and `io` reference each other, but the import order is one-way at
  load time. `core.py` imports `io/aavso.py` (for `write_aavso_extended`) at the
  top of the file, and `io/aavso.py` does **not** import `core.py`. The only
  `io` module that imports `core.py` is `io/tess.py`, which is exposed lazily by
  `io/__init__.py` (a module-level `__getattr__`), so it is loaded only on first
  use — after `core.py` has finished importing. This replaces the previous
  in-method lazy import that worked around a genuine `core ↔ io` cycle.

## Core + pipeline cluster (file level)

The photometry pipeline, differential photometry, and calibration utilities
all sit on top of the core data layer.

*Arrows: **solid** = an import (importer → imported); **dashed** = a notebook driving a UI layer, or a looser link.*

```mermaid
flowchart LR
    classDef external fill:#eceff1,stroke:#90a4ae,stroke-dasharray: 5 5,color:#212121

    subgraph sg_phot["stellarphot.photometry"]
        direction TB
        phot_py["photometry.py<br/>AperturePhotometry"]
        srcdet_py["source_detection.py<br/>source_detection, compute_fwhm"]
        prof_py["profiles.py<br/>CenterAndProfile, find_center"]
    end

    subgraph sg_diff["stellarphot.differential_photometry"]
        direction TB
        relflux_py["aij_rel_fluxes.py<br/>calc_aij_relative_flux"]
        vsxmag_py["vsx_mags.py<br/>calc_vmag, calc_multi_vmag"]
    end

    subgraph sg_utils["stellarphot.utils"]
        direction TB
        magtrans_py["magnitude_transforms.py"]
        magsys_py["magnitude_system_transforms.py"]
        computil_py["comparison_utils.py"]
        vermig_py["version_migrator.py"]
    end

    subgraph sg_core["core data layer"]
        direction TB
        core_py["core.py<br/>PhotometryData, CatalogData,<br/>SourceListData"]
        catalogs_py["catalogs.py<br/>apass_dr9, vsx_vizier, refcat2"]
        tabrep_py["table_representations.py"]
    end

    settings_ext["stellarphot.settings"]:::external

    phot_py -->|"compute_fwhm,<br/>fast_fwhm_from_image"| srcdet_py
    prof_py -->|"calculate_noise"| phot_py
    phot_py --> core_py
    phot_py --> settings_ext
    srcdet_py --> core_py
    srcdet_py --> settings_ext
    relflux_py --> core_py
    magtrans_py -->|"apass_dr9, refcat2"| catalogs_py
    magtrans_py --> magsys_py
    computil_py -->|"apass_dr9, vsx_vizier"| catalogs_py
    vermig_py --> core_py
    vermig_py --> settings_ext
    core_py --> settings_ext
    core_py --> tabrep_py
    catalogs_py -->|"CatalogData"| core_py
    catalogs_py -->|"PassbandMap"| settings_ext
    tabrep_py --> settings_ext

    click phot_py href "../stellarphot/photometry/photometry.py" "photometry.py"
    click srcdet_py href "../stellarphot/photometry/source_detection.py" "source_detection.py"
    click prof_py href "../stellarphot/photometry/profiles.py" "profiles.py"
    click relflux_py href "../stellarphot/differential_photometry/aij_rel_fluxes.py" "aij_rel_fluxes.py"
    click vsxmag_py href "../stellarphot/differential_photometry/vsx_mags.py" "vsx_mags.py"
    click magtrans_py href "../stellarphot/utils/magnitude_transforms.py" "magnitude_transforms.py"
    click magsys_py href "../stellarphot/utils/magnitude_system_transforms.py" "magnitude_system_transforms.py"
    click computil_py href "../stellarphot/utils/comparison_utils.py" "comparison_utils.py"
    click vermig_py href "../stellarphot/utils/version_migrator.py" "version_migrator.py"
    click core_py href "../stellarphot/core.py" "core.py"
    click catalogs_py href "../stellarphot/catalogs.py" "catalogs.py"
    click tabrep_py href "../stellarphot/table_representations.py" "table_representations.py"
```

Key contents of each file:

- [`core.py`](../stellarphot/core.py) — `BaseEnhancedTable` (validated `QTable`) and its subclasses
  `PhotometryData`, `CatalogData`, `SourceListData` (table data structures only).
- [`catalogs.py`](../stellarphot/catalogs.py) — catalog-fetcher functions
  `apass_dr9()`, `refcat2()`, `vsx_vizier()` that build a `CatalogData` from
  Vizier/astroquery. Still importable from `stellarphot.core` via a deprecated shim.
- [`table_representations.py`](../stellarphot/table_representations.py) — `generate_table_representers()`,
  `serialize_models_in_table_meta()`, `deserialize_models_in_table_meta()`
  (round-trips pydantic settings models stored in table metadata).
- [`photometry/photometry.py`](../stellarphot/photometry/photometry.py) — `AperturePhotometry`,
  `single_image_photometry()`, `multi_image_photometry()`,
  `find_too_close()`, `clipped_sky_per_pix_stats()`, `calculate_noise()`.
- [`photometry/source_detection.py`](../stellarphot/photometry/source_detection.py) — `source_detection()`,
  `compute_fwhm()`, `fast_fwhm_from_image()`.
- [`photometry/profiles.py`](../stellarphot/photometry/profiles.py) — `find_center()`, `CenterAndProfile`
  (radial profile, curve of growth, SNR).
- [`differential_photometry/aij_rel_fluxes.py`](../stellarphot/differential_photometry/aij_rel_fluxes.py) — `calc_aij_relative_flux()`,
  `add_relative_flux_column()`, `add_in_quadrature()`.
- [`differential_photometry/vsx_mags.py`](../stellarphot/differential_photometry/vsx_mags.py) — `calc_vmag()`, `calc_multi_vmag()`.
- [`utils/magnitude_transforms.py`](../stellarphot/utils/magnitude_transforms.py) — `transform_to_catalog()`,
  `calculate_transform_coefficients()`, `transform_magnitudes()`,
  `filter_transform()`.
- [`utils/magnitude_system_transforms.py`](../stellarphot/utils/magnitude_system_transforms.py) — `PanStarrs1ToJohnsonCousins`,
  `USNOPrimeToSDSSDR7`, `transform_apass_bands()`, `transform_refcat2_bands()`.
- [`utils/comparison_utils.py`](../stellarphot/utils/comparison_utils.py) — `set_up()`, `crossmatch_APASS2VSX()`,
  `mag_scale()`, `in_field()`, `read_file()`.
- [`utils/version_migrator.py`](../stellarphot/utils/version_migrator.py) — `VersionMigrator` (stellarphot 1 → 2 data).

## UI cluster (gui + settings, file level)

The entire widget layer now lives in `stellarphot.gui`: the auto-generated
settings forms (`views.py`, `ui_generator`), the settings widgets
(`custom_widgets.py`), the file chooser (`fits_opener.py`), and the
seeing-profile/comparison-star widgets. It is built on top of the pure-Pydantic
`stellarphot.settings` package (`models.py`, `settings_files.py`).

`stellarphot.settings` is **data-only**: importing it (and hence
`import stellarphot` and the core data layer) loads only the pydantic models and
file handling, never the GUI stack. The GUI libraries
(`ipywidgets`/`ipyautoui`/`astrowidgets`/`ginga`) are confined to
`stellarphot.gui` and ship only with the optional `[gui]` extra, so a base
install stays headless; importing `stellarphot.gui` is what pulls them in.

*Arrows: **solid** = an import (importer → imported); **dashed** = a notebook driving a UI layer, or a looser link.*

```mermaid
flowchart LR
    classDef external fill:#eceff1,stroke:#90a4ae,stroke-dasharray: 5 5,color:#212121
    classDef nbstyle fill:#fff8e1,stroke:#f9a825,color:#212121

    nb["notebooks/<br/>(launcher)"]:::nbstyle

    subgraph sg_gui["stellarphot.gui (the [gui] extra)"]
        direction TB
        pac_py["profile_and_comps.py<br/>ComparisonAndSeeing"]
        comp_py["comparison_functions.py<br/>ComparisonViewer"]
        seeing_py["seeing_profile_functions.py<br/>SeeingProfileWidget"]
        photwidg_py["photometry_widget_functions.py<br/>TessAnalysisInputControls"]
        cw_py["custom_widgets.py<br/>ReviewSettings, ChooseOrMakeNew,<br/>PhotometryRunner, TessPhotometrySetup"]
        views_py["views.py<br/>ui_generator"]
        fo_py["fits_opener.py<br/>FitsOpener"]
    end

    subgraph sg_set["stellarphot.settings (pure Pydantic)"]
        direction TB
        sf_py["settings_files.py<br/>SavedSettings,<br/>PhotometryWorkingDirSettings"]
        models_py["models.py<br/>PhotometrySettings, Camera,<br/>Observatory, ..."]
        ap_py["astropy_pydantic.py"]
        aavsom_py["aavso_models.py"]
        aavsos_py["aavso_submission.py"]
    end

    core_ext["core.py"]:::external
    phot_ext["photometry/"]:::external
    plot_ext["plotting/"]:::external
    io_ext["io/"]:::external
    utils_ext["utils/"]:::external

    nb -.-> pac_py
    nb -.-> photwidg_py
    nb -.-> cw_py

    pac_py --> comp_py
    pac_py --> seeing_py
    comp_py -->|"set_keybindings"| seeing_py
    comp_py --> cw_py
    comp_py --> fo_py
    comp_py --> core_ext
    comp_py --> utils_ext
    seeing_py --> cw_py
    seeing_py --> fo_py
    seeing_py -->|"CenterAndProfile"| phot_ext
    seeing_py -->|"seeing_plot"| plot_ext
    seeing_py -->|"TessSubmission"| io_ext
    photwidg_py --> cw_py
    photwidg_py --> core_ext

    cw_py --> models_py
    cw_py --> sf_py
    cw_py -->|"ui_generator"| views_py
    cw_py --> fo_py
    cw_py -->|"tess_photometry_setup"| io_ext
    sf_py --> models_py
    views_py --> models_py
    models_py --> ap_py
    models_py --> aavsom_py
    aavsos_py --> models_py

    click nb href "../stellarphot/notebooks/" "notebooks/"
    click pac_py href "../stellarphot/gui/profile_and_comps.py" "profile_and_comps.py"
    click comp_py href "../stellarphot/gui/comparison_functions.py" "comparison_functions.py"
    click seeing_py href "../stellarphot/gui/seeing_profile_functions.py" "seeing_profile_functions.py"
    click photwidg_py href "../stellarphot/gui/photometry_widget_functions.py" "photometry_widget_functions.py"
    click cw_py href "../stellarphot/gui/custom_widgets.py" "custom_widgets.py"
    click views_py href "../stellarphot/gui/views.py" "views.py"
    click sf_py href "../stellarphot/settings/settings_files.py" "settings_files.py"
    click models_py href "../stellarphot/settings/models.py" "models.py"
    click fo_py href "../stellarphot/gui/fits_opener.py" "fits_opener.py"
    click ap_py href "../stellarphot/settings/astropy_pydantic.py" "astropy_pydantic.py"
    click aavsom_py href "../stellarphot/settings/aavso_models.py" "aavso_models.py"
    click aavsos_py href "../stellarphot/settings/aavso_submission.py" "aavso_submission.py"
```

Key contents of each file:

- [`settings/models.py`](../stellarphot/settings/models.py) — pydantic models: `Camera`, `Observatory`,
  `PhotometryApertures`, `SourceLocationSettings`,
  `PhotometryOptionalSettings`, `PassbandMap`/`PassbandMapEntry`,
  `LoggingSettings`, `PhotometrySettings` (the aggregate), `Exoplanet`,
  `PhotometryRunSettings`, `PartialPhotometrySettings`.
- [`settings/astropy_pydantic.py`](../stellarphot/settings/astropy_pydantic.py) — pydantic validators for astropy types
  (`UnitType`, `QuantityType`, `EquivalentTo`, `WithPhysicalType`,
  `AstropyValidator`).
- [`settings/settings_files.py`](../stellarphot/settings/settings_files.py) — `SavedSettings` (per-user storage of
  cameras/observatories/passband maps), `PhotometryWorkingDirSettings`
  (loads/saves `photometry_settings.json` in the working directory).
- [`gui/views.py`](../stellarphot/gui/views.py) — `ui_generator()` (builds an `ipyautoui` widget from
  any pydantic model).
- [`gui/custom_widgets.py`](../stellarphot/gui/custom_widgets.py) — `ChooseOrMakeNew`, `Confirm`,
  `SettingWithTitle`, `ReviewSettings`, `PhotometryRunner`,
  `TessPhotometrySetup`, `Spinner`.
- [`gui/fits_opener.py`](../stellarphot/gui/fits_opener.py) — `FitsOpener` (file chooser + lazy
  `CCDData`/header access).
- [`gui/seeing_profile_functions.py`](../stellarphot/gui/seeing_profile_functions.py) — `SeeingProfileWidget`,
  `set_keybindings()`.
- [`gui/comparison_functions.py`](../stellarphot/gui/comparison_functions.py) — `ComparisonViewer`,
  `make_markers()`.
- [`gui/photometry_widget_functions.py`](../stellarphot/gui/photometry_widget_functions.py) — `TessAnalysisInputControls`,
  `filter_by_dates()`.
- [`gui/profile_and_comps.py`](../stellarphot/gui/profile_and_comps.py) — `ComparisonAndSeeing` (combines the
  seeing and comparison widgets).

## Output / IO cluster (file level)

*Arrows: **solid** = an import (importer → imported); **dashed** = a notebook driving a UI layer, or a looser link.*

```mermaid
flowchart LR
    classDef external fill:#eceff1,stroke:#90a4ae,stroke-dasharray: 5 5,color:#212121

    subgraph sg_io["stellarphot.io"]
        direction TB
        aij_py["aij.py<br/>ApertureFileAIJ, parse_aij_table"]
        aavso_py["aavso.py<br/>write_aavso_extended"]
        tess_py["tess.py<br/>TOI, TessSubmission,<br/>TessTargetFile"]
    end

    subgraph sg_tf["stellarphot.transit_fitting"]
        direction TB
        tf_core["core.py<br/>TransitModelFit"]
        tf_io["io.py<br/>get_tic_info"]
        tf_plot["plotting.py<br/>plot_predict_ingress_egress"]
    end

    tf_gui["gui/transit_fitting_gui.py<br/>exotic_settings_widget<br/>(stellarphot.gui)"]:::external

    subgraph sg_plot["stellarphot.plotting"]
        direction TB
        aijplots_py["aij_plots.py<br/>seeing_plot"]
        transitplots_py["transit_plots.py<br/>plot_transit_lightcurve"]
        multinight_py["multi_night_plots.py<br/>multi_night"]
    end

    core_ext["core.py"]:::external
    settings_ext["settings/"]:::external
    batman_ext["pytransit<br/>(transit models)"]:::external
    mast_ext["astroquery MAST"]:::external

    tess_py -->|"get_tic_info"| tf_io
    tess_py --> core_ext
    tess_py --> settings_ext
    aavso_py -->|"AAVSOFilters,<br/>AAVSOSubmissionHeader"| settings_ext
    tf_gui --> tf_io
    tf_io -.-> mast_ext
    tf_core -.-> batman_ext
    aijplots_py -->|"PhotometryApertures"| settings_ext

    click aij_py href "../stellarphot/io/aij.py" "io/aij.py"
    click aavso_py href "../stellarphot/io/aavso.py" "io/aavso.py"
    click tess_py href "../stellarphot/io/tess.py" "io/tess.py"
    click tf_core href "../stellarphot/transit_fitting/core.py" "transit_fitting/core.py"
    click tf_gui href "../stellarphot/gui/transit_fitting_gui.py" "gui/transit_fitting_gui.py"
    click tf_io href "../stellarphot/transit_fitting/io.py" "transit_fitting/io.py"
    click tf_plot href "../stellarphot/transit_fitting/plotting.py" "transit_fitting/plotting.py"
    click aijplots_py href "../stellarphot/plotting/aij_plots.py" "plotting/aij_plots.py"
    click transitplots_py href "../stellarphot/plotting/transit_plots.py" "plotting/transit_plots.py"
    click multinight_py href "../stellarphot/plotting/multi_night_plots.py" "plotting/multi_night_plots.py"
```

Key contents of each file:

- [`io/aij.py`](../stellarphot/io/aij.py) — `ApertureAIJ`, `MultiApertureAIJ`, `ApertureFileAIJ`,
  `Star`, `generate_aij_table()`, `parse_aij_table()` (AstroImageJ
  compatibility).
- [`io/aavso.py`](../stellarphot/io/aavso.py) — `write_aavso_extended()` plus field
  validators/formatters for the AAVSO extended file format.
- [`io/tess.py`](../stellarphot/io/tess.py) — `TessSubmission` (TFOP file naming), `TOI` (transit
  parameters fetched by TIC ID), `TessTargetFile` (nearby GAIA sources),
  `tess_photometry_setup()`.
- [`transit_fitting/core.py`](../stellarphot/transit_fitting/core.py) — `TransitModelFit`, `TransitModelOptions`.
- [`gui/transit_fitting_gui.py`](../stellarphot/gui/transit_fitting_gui.py) — EXOTIC settings widget and TIC/TOI
  population helpers.
- [`transit_fitting/io.py`](../stellarphot/transit_fitting/io.py) — `get_tic_info()` (MAST catalog query).
- [`transit_fitting/plotting.py`](../stellarphot/transit_fitting/plotting.py) — `plot_predict_ingress_egress()`.
- [`plotting/aij_plots.py`](../stellarphot/plotting/aij_plots.py) — `seeing_plot()`.
- [`plotting/transit_plots.py`](../stellarphot/plotting/transit_plots.py) — `plot_transit_lightcurve()`,
  `plot_many_factors()`, `bin_data()`, `scale_and_shift()`.
- [`plotting/multi_night_plots.py`](../stellarphot/plotting/multi_night_plots.py) — `plot_magnitudes()`, `multi_night()`.

## External dependencies (major ones per component)

| Component | Major external dependencies |
|---|---|
| Core data layer | astropy (QTable, SkyCoord, Time, units), astroquery (Vizier, XMatch), pandas, lightkurve — no GUI stack at import time |
| Photometry engine | photutils (apertures, DAOStarFinder, centroids, profiles), astropy, ccdproc |
| Settings | pydantic only — no GUI dependencies (the widget layer moved to `stellarphot.gui`) |
| Transit fitting | pytransit, astropy.modeling, scipy, astroquery (MAST) |
| GUI layer | the optional `[gui]` extra: ipywidgets, ipyautoui, ipyfilechooser, astrowidgets, ginga, jupyter-app-launcher, papermill (plus matplotlib from the base install) — not installed by a base `pip install stellarphot` |
| Plotting | matplotlib |
