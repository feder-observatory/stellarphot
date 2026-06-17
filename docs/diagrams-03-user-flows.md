# stellarphot — Example User Flows

These flowcharts show how a typical user moves through stellarphot, from
configuring their equipment to submitting results.

**Legend** — the node types used on this page:

```mermaid
flowchart LR
    classDef action fill:#e3f2fd,stroke:#1565c0,color:#212121
    classDef widget fill:#fff3e0,stroke:#ef6c00,color:#212121
    classDef artifact fill:#eceff1,stroke:#546e7a,color:#212121
    classDef external fill:#f3e5f5,stroke:#7b1fa2,color:#212121

    l_action(["A user action"]):::action
    l_widget["A stellarphot widget<br/>or function"]:::widget
    l_artifact[("A file in the<br/>working directory")]:::artifact
    l_external["An external service<br/>or web query"]:::external

    l_action ~~~ l_widget ~~~ l_artifact ~~~ l_external
```

- **Solid arrows** — progression from one step to the next.
- **Dashed arrows** — file reads and writes.

## Flow 1 — Main photometry workflow (launcher notebooks 1 → 4)

A first-time user configures their camera and observatory once (saved
per-user), then for each night of images: inspects a star's seeing profile to
pick apertures, selects comparison stars, reviews the combined settings, and
runs photometry on the whole folder.

*Arrows: **solid** = progression from one step to the next; **dashed** = file reads and writes.*

```mermaid
flowchart LR
    classDef action fill:#e3f2fd,stroke:#1565c0,color:#212121
    classDef widget fill:#fff3e0,stroke:#ef6c00,color:#212121
    classDef artifact fill:#eceff1,stroke:#546e7a,color:#212121

    start(["Start JupyterLab,<br/>open launcher"]):::action

    subgraph sg_nb1["1 - Saveable settings"]
        direction TB
        w_review1["ReviewSettings:<br/>Camera, Observatory,<br/>PassbandMap"]:::widget
    end

    subgraph sg_nb2["2 - Seeing profile and comparison stars"]
        direction TB
        a_open["Open a FITS image,<br/>click on target star"]:::action
        w_seeing["SeeingProfileWidget:<br/>pick aperture radius from<br/>radial profile and SNR"]:::widget
        w_comp["ComparisonViewer:<br/>select comparison stars<br/>(APASS, VSX overlays)"]:::widget
    end

    subgraph sg_nb3["3 - Review settings"]
        direction TB
        w_review3["ReviewSettings:<br/>confirm all seven<br/>settings groups"]:::widget
    end

    subgraph sg_nb4["4 - Launch photometry"]
        direction TB
        a_pick["Choose one image of<br/>the object of interest"]:::action
        w_runner["PhotometryRunner:<br/>papermill executes<br/>photometry_runner.ipynb"]:::widget
        w_apphot["AperturePhotometry on<br/>every image in folder"]:::widget
    end

    saved[("per-user saved settings<br/>(cameras, observatories,<br/>passband maps)")]:::artifact
    partial[("partial_photometry_settings.json")]:::artifact
    srclist[("source_locations.ecsv")]:::artifact
    full[("photometry_settings.json")]:::artifact
    photecsv[("photometry.ecsv<br/>(PhotometryData)")]:::artifact

    start --> w_review1
    w_review1 -.->|"save"| saved
    start --> a_open
    a_open --> w_seeing
    w_seeing --> w_comp
    w_seeing -.->|"aperture settings"| partial
    w_comp -.->|"writes"| srclist
    w_comp -.->|"source list path"| partial
    w_review3 -.->|"reads"| partial
    saved -.->|"choose camera /<br/>observatory"| w_review3
    w_review3 -.->|"saves complete"| full
    a_pick --> w_runner
    w_runner --> w_apphot
    full -.->|"reads"| w_apphot
    srclist -.->|"reads"| w_apphot
    w_apphot -.->|"writes"| photecsv

    click w_review1 href "../stellarphot/gui/custom_widgets.py" "custom_widgets.py"
    click w_seeing href "../stellarphot/gui/seeing_profile_functions.py" "seeing_profile_functions.py"
    click w_comp href "../stellarphot/gui/comparison_functions.py" "comparison_functions.py"
    click w_review3 href "../stellarphot/gui/custom_widgets.py" "custom_widgets.py"
    click w_runner href "../stellarphot/gui/custom_widgets.py" "custom_widgets.py"
    click w_apphot href "../stellarphot/photometry/photometry.py" "photometry.py"
```

*Source: [seeing_profile_functions.py](../stellarphot/gui/seeing_profile_functions.py), [comparison_functions.py](../stellarphot/gui/comparison_functions.py), [custom_widgets.py](../stellarphot/gui/custom_widgets.py), [photometry.py](../stellarphot/photometry/photometry.py)*

## Flow 2 — Exoplanet transit analysis

Starting from the photometry table produced by Flow 1, the user computes
relative fluxes, then fits transit models of increasing sophistication and
prepares submission files.

*Arrows: **solid** = progression from one step to the next; **dashed** = file reads and writes.*

```mermaid
flowchart LR
    classDef action fill:#e3f2fd,stroke:#1565c0,color:#212121
    classDef widget fill:#fff3e0,stroke:#ef6c00,color:#212121
    classDef artifact fill:#eceff1,stroke:#546e7a,color:#212121

    photecsv[("photometry.ecsv")]:::artifact

    subgraph sg_flux["Calculate relative flux"]
        direction TB
        w_flux["add_relative_flux_column():<br/>ensemble of comparison stars"]:::widget
    end

    relflux[("photometry-relative-flux.ecsv")]:::artifact
    ticinfo[("TIC-(id)-info.json")]:::artifact

    subgraph sg_fit1["Initial exoplanet model"]
        direction TB
        w_inputs1["TessAnalysisInputControls:<br/>choose photometry file,<br/>TIC info, passband"]:::widget
        w_fit1["TransitModelFit.setup_model()<br/>then .fit()"]:::widget
        w_plot1["plot_transit_lightcurve()"]:::widget
    end

    subgraph sg_fit2["Second exoplanet model fit"]
        direction TB
        w_fit2["TransitModelFit with<br/>detrending (airmass,<br/>width, sky) via<br/>TransitModelOptions"]:::widget
    end

    subgraph sg_exotic["Exoplanet model fit with EXOTIC"]
        direction TB
        w_exotic["exotic_settings_widget()<br/>populate_TOI_boxes()"]:::widget
        a_exotic["Run EXOTIC with<br/>generated settings"]:::action
    end

    exoticjson[("EXOTIC settings .json")]:::artifact
    aavso["PhotometryData<br/>.write_aavso_extended()"]:::widget
    aavsofile[("AAVSO extended<br/>format file")]:::artifact

    photecsv -.-> w_flux
    w_flux -.->|"writes"| relflux
    relflux -.-> w_inputs1
    ticinfo -.-> w_inputs1
    w_inputs1 --> w_fit1
    w_fit1 --> w_plot1
    w_plot1 --> w_fit2
    w_fit2 --> w_exotic
    w_exotic -.->|"writes"| exoticjson
    exoticjson -.-> a_exotic
    relflux -.-> aavso
    aavso -.->|"writes"| aavsofile

    click w_flux href "../stellarphot/differential_photometry/aij_rel_fluxes.py" "aij_rel_fluxes.py"
    click w_inputs1 href "../stellarphot/gui/photometry_widget_functions.py" "photometry_widget_functions.py"
    click w_fit1 href "../stellarphot/transit_fitting/core.py" "transit_fitting/core.py"
    click w_plot1 href "../stellarphot/plotting/transit_plots.py" "transit_plots.py"
    click w_fit2 href "../stellarphot/transit_fitting/core.py" "transit_fitting/core.py"
    click w_exotic href "../stellarphot/gui/transit_fitting_gui.py" "gui/transit_fitting_gui.py"
    click aavso href "../stellarphot/core.py" "core.py"
```

*Source: [aij_rel_fluxes.py](../stellarphot/differential_photometry/aij_rel_fluxes.py), [photometry_widget_functions.py](../stellarphot/gui/photometry_widget_functions.py), [transit_fitting/core.py](../stellarphot/transit_fitting/core.py), [transit_plots.py](../stellarphot/plotting/transit_plots.py), [gui/transit_fitting_gui.py](../stellarphot/gui/transit_fitting_gui.py)*

## Flow 3 — Supporting flows

### Generate a TESS target source list

Run before Flow 1 when observing a TESS Object of Interest: it fetches the
transit parameters and nearby GAIA sources so the photometry includes the
stars TFOP wants checked.

*Arrows: **solid** = progression from one step to the next; **dashed** = file reads and writes.*

```mermaid
flowchart LR
    classDef action fill:#e3f2fd,stroke:#1565c0,color:#212121
    classDef widget fill:#fff3e0,stroke:#ef6c00,color:#212121
    classDef artifact fill:#eceff1,stroke:#546e7a,color:#212121
    classDef external fill:#f3e5f5,stroke:#7b1fa2,color:#212121

    a_tic["Enter TIC ID or choose<br/>an image with one<br/>in its header"]:::action
    w_tps["TessPhotometrySetup"]:::widget
    f_setup["tess_photometry_setup()"]:::widget
    mast["MAST / ExoFOP<br/>(TOI.from_tic_id)"]:::external
    gaia["GAIA target list service<br/>(TessTargetFile)"]:::external
    info[("TIC-(id)-info.json")]:::artifact
    srclist[("TIC-(id)-source-list-input.ecsv")]:::artifact

    a_tic --> w_tps
    w_tps --> f_setup
    f_setup --> mast
    f_setup --> gaia
    f_setup -.->|"writes"| info
    f_setup -.->|"writes"| srclist

    click w_tps href "../stellarphot/gui/custom_widgets.py" "custom_widgets.py"
    click f_setup href "../stellarphot/io/tess.py" "io/tess.py"
```

*Source: [custom_widgets.py](../stellarphot/gui/custom_widgets.py), [io/tess.py](../stellarphot/io/tess.py)*

### Calibrate magnitudes to a catalog (APASS DR9)

Run after Flow 1 for variable-star work: instrumental magnitudes are
transformed onto the catalog system, night by night.

*Arrows: **solid** = progression from one step to the next; **dashed** = file reads and writes.*

```mermaid
flowchart LR
    classDef action fill:#e3f2fd,stroke:#1565c0,color:#212121
    classDef widget fill:#fff3e0,stroke:#ef6c00,color:#212121
    classDef artifact fill:#eceff1,stroke:#546e7a,color:#212121
    classDef external fill:#f3e5f5,stroke:#7b1fa2,color:#212121

    photecsv[("photometry.ecsv")]:::artifact
    a_load["Open 'Calibrate<br/>magnitudes' notebook"]:::action
    w_t2c["transform_to_catalog()"]:::widget
    apass["APASS DR9 / RefCat2<br/>via apass_dr9(), refcat2()"]:::external
    w_fit["curve_fit of<br/>calibrated_from_instrumental()"]:::widget
    calecsv[("photometry with<br/>calibrated mag columns")]:::artifact
    vsx["vsx_vizier() marks known<br/>variable stars"]:::external

    a_load -.->|"reads"| photecsv
    a_load --> w_t2c
    w_t2c --> apass
    w_t2c --> w_fit
    w_t2c --> vsx
    w_t2c -.->|"writes"| calecsv

    click w_t2c href "../stellarphot/utils/magnitude_transforms.py" "magnitude_transforms.py"
    click w_fit href "../stellarphot/utils/magnitude_transforms.py" "magnitude_transforms.py"
```

*Source: [magnitude_transforms.py](../stellarphot/utils/magnitude_transforms.py)*
