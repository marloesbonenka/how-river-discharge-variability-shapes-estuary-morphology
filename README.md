# How River Discharge Variability Shapes Estuary Morphology

This repository contains the reproducible analysis workflow for the PhD research project **“How River Discharge Variability Shapes Estuary Morphology.”** The code investigates how variability in river discharge influences estuarine morphology using numerical modelling, post-processing, and quantitative analysis.

The repository is organised to connect model setup, post-processing, derived data, and figure-generation scripts directly to the analyses and figures presented in the associated paper.
It provides the scripts required to reproduce the main analyses and figures of the paper. The general workflow is:

```text
model setup
    ↓
numerical model runs
    ↓
post-processing
    ↓
derived metrics / diagnostics
    ↓
figure generation
    ↓
paper figures and results
```

### 1. Set up the model

The `model_setup/` directory contains the files and scripts used to define model experiments and generate the discharge-variability parameters used in the simulations.

This includes:

* MDU templates and model configuration files
* Generation of `R_peak` values
* Generation of `n_peaks` values
* Generation of discharge (`Q`) parameters
* Other experiment-specific setup files

### 2. Post-process model output

The `postprocessing/` directory contains scripts for reading, processing, and diagnosing model output.

Examples include:

* `F_map_cache` — caching or preparing model-derived maps
* `F_loaddata` — loading model output for analysis
* Diagnostic scripts for checking model behaviour and derived quantities
* Calculation of summary statistics used in subsequent analysis

### 3. Generate derived data

Small processed datasets required for analysis are stored in:

```text
data/derived/
```

These may include quantities such as:

* Discharge percentiles
* `L_TI` time series
* Other processed model outputs and summary statistics

Large raw model outputs are not stored in this repository unless otherwise stated.

### 4. Reproduce paper figures

The `figures/` directory contains the plotting scripts used to generate the figures for the paper.

Where possible, there is **one plotting script per paper figure**, making it straightforward to identify which code produced each figure.

The plotting scripts follow **AGU formatting conventions** where applicable.

For example:

```text
figures/
├── figure_01_*.py
├── figure_02_*.py
├── figure_03_*.py
└── ...
```

The exact filenames and figure-to-script mapping are documented in the repository.

### Reproducing a figure

From the repository root:

```bash
conda env create -f environment.yml
conda activate <environment-name>
```

Then run the relevant post-processing and figure script:

```bash
python postprocessing/<script-name>.py
python figures/<figure-script-name>.py
```

> **Note:** Exact commands for reproducing individual figures will be added as the analysis workflow is finalised.

---

## Contact Information

For questions about the research, model setup, analysis, or code, please contact:

* **Author:** Marloes Bonenkamp
* **Email:** m.bonenkamp@tudelft.nl
* **ORCID:** https://orcid.org/0009-0009-1292-2444
* **Institution:** Delft University of Technology, The Netherlands

For code issues or reproducibility problems, please open an issue in this repository.

---

## Install Instructions

The computational environment is defined in `environment.yml` and uses packages from the **conda-forge** ecosystem.

### Requirements

* Conda or Miniconda
* The environment specified in `environment.yml`
* <Additional software/model requirements>

### Create the environment

From the repository root:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate <environment-name>
```

### Verify the installation

<Verification command / test>

The repository assumes that the required numerical modelling software and dependencies are available within the environment or are otherwise configured as described in the model documentation.

<Additional model installation instructions>

---

## Citation Instructions

If you use the code, analysis workflow, or derived results from this repository, please cite the associated paper:

> <Authors>. (<Year>). *How River Discharge Variability Shapes Estuary Morphology*. <Journal>. <DOI>

Please also cite the repository using the information provided in `CITATION.cff`.

### Repository citation

> <Authors>. (<Year>). *How River Discharge Variability Shapes Estuary Morphology — Research Code*. <Repository>. <DOI>

**DOI:** <DOI>

The `CITATION.cff` file contains the machine-readable citation information for this repository.

---

## Contribution Statement

This repository primarily contains research code supporting a specific scientific publication. Contributions that improve reproducibility, documentation, robustness, or code quality are welcome.

To contribute:

1. Open an issue describing the proposed change or problem.
2. Where relevant, include a minimal example demonstrating the issue.
3. Submit a pull request with a clear description of the changes.
4. For changes that affect scientific results, clearly document the methodological implications.

Please avoid modifying derived results or figure-generation workflows without documenting how the change affects the published analysis.

---

## Reference Material

The main reference for the scientific methodology is the associated paper:

> **How River Discharge Variability Shapes Estuary Morphology**

<Authors>. <Year>. <Journal>. <DOI>

Additional documentation:

* **Paper:** <paper-link>
* **Supplementary material:** <supplementary-material-link>
* **Model documentation:** <model-documentation-link>
* **Data repository:** <data-repository-link>
* **Documentation:** <documentation-link>

### Repository structure

```text
.
├── README.md
│   # Overview of the repository and mapping between scripts and paper figures/sections
│
├── environment.yml
│   # Conda-forge computational environment
│
├── model_setup/
│   # Model configuration and experiment generation
│   ├── MDU templates
│   ├── R_peak generation
│   ├── n_peaks generation
│   └── Q parameter generation
│
├── postprocessing/
│   # Model-output loading, processing, and diagnostics
│   ├── F_map_cache
│   ├── F_loaddata
│   └── diagnostic scripts
│
├── figures/
│   # AGU-formatted scripts for generating paper figures
│   └── one script per figure where possible
│
├── data/
│   └── derived/
│       # Small processed outputs used by the analysis
│       ├── percentiles
│       ├── L_TI series
│       └── other derived quantities
│
├── LICENSE
│   # Software license
│
└── CITATION.cff
    # Citation metadata
```

### Mapping code to the paper

The README is intended to provide a direct link between the computational workflow and the paper. The following table should be updated as the analysis is finalised:

| Paper component           | Repository location  | Description                                                 |
| ------------------------- | -------------------- | ----------------------------------------------------------- |
| Model setup / Methods     | `model_setup/`       | Generation of model configurations and discharge parameters |
| Post-processing / Methods | `postprocessing/`    | Loading and processing model output                         |
| Derived metrics           | `data/derived/`      | Small processed datasets used for analysis                  |
| Figure 1                  | `figures/`           | <script name>                                               |
| Figure 2                  | `figures/`           | <script name>                                               |
| Figure 3                  | `figures/`           | <script name>                                               |
| Figure 4                  | `figures/`           | <script name>                                               |
| <Section>                 | `<directory/script>` | <description>                                               |

---

## Licensing Statement

The code in this repository is released under the license specified in `LICENSE`.

<License name and additional licensing information>

Please note that the license for the code does not necessarily apply to external datasets, model outputs, or other third-party materials referenced by the project.

---

## Acknowledgments

This research was supported by:

* **Funder:** <funder name>
* **Grant/project number:** <grant number>
* **Institution:** <institution>
* **Research group:** <research group>

We also acknowledge:

* <Collaborator / supervisor>
* <Research group / institution>
* <Data provider>
* <Computational infrastructure / HPC facility>
* <Model developers>

for their contributions to this research.

---

## Related Publication

**How River Discharge Variability Shapes Estuary Morphology**

<Authors>. <Year>. <Journal>. <DOI>

---

## Reproducibility

This repository is designed to make the computational workflow underlying the paper transparent and reproducible.

The repository distinguishes between:

* **Model setup** — how experiments and discharge scenarios are defined;
* **Post-processing** — how model output is loaded, processed, and diagnosed;
* **Derived data** — small processed datasets required for analysis;
* **Figure scripts** — code used to generate the figures presented in the paper.

Large model outputs are not included unless explicitly stated. Where external data or model outputs are required, their source and access instructions should be documented here:

<Instructions for accessing external model output / datasets>

---
