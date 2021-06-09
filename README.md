# Information
- Python v3.8.6
- Ubuntu 20.04.2 LTS - 2x Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz (12 Cores)
- List of third party libraries is in "code/requirements.txt"  
Run `pip install -r requirements.txt` (or equivalent command for conda) to install required version of the libraries
- All figures used in cyton2 paper are pre-generated in respective output folders e.g. FigX-name. Note that by default the codes will save the outputs in corresponding "out" folder and will replace any existing files.

# Section 1: Filming data
```bash
code
└── tracking
    ├── data
    │   ├── _processed
    │   │   ├── _parse
    │   │   └── collapsed_times
    │   ├── b_cpg
    │   │   ├── cpg3
    │   │   └── cpg4
    │   ├── t_il2
    │   │   ├── 20131218
    │   │   └── 20140121
    │   └── t_misc
    │       ├── 20140211
    │       └── 20140325
    └── out
        ├── Fig2-1-clones
        ├── Fig2-2-raw
        ├── Fig2-3-corr
        ├── Fig3-Simulation
        └── Fig4-dist
```
Data names:
- b_cpg/cpg3 (B-exp1)
- b_cpg/cpg4 (B-exp2)
- t_il2/20131218 + t_il2/20140121 (1U, 3U and 10U IL-2): CD8+ T cells aggregated dataset
- t_misc/20140211 (T-exp1)
- t_misc/20140325 (T-exp2)

Correspond to Figure 2, 3 and 4 in the paper. 
- Fig2-1-clones: Clonal collapse
- Fig2-2-raw: Cascade plot
- Fig2-3-corr: Correlation pair-plot
- Fig3-Simulation: Agent-based Model
- Fig4-dist: Best parametric distribution class for measured times

# Section 2: Cyton2 model
```bash
code
└── cyton
    ├── data
    ├── out
    │   ├── FigS7-Simulation
    │   ├── _lognormal
    │   │   ├── indiv
    │   │   │   └── Fig5
    │   │   │       ├── FigAB-allReps
    │   │   │       │   └── fitPlots
    │   │   │       └── FigC-rmvTPS
    │   │   │           ├── data
    │   │   │           └── fitPlots
    │   │   └── joint
    │   └── _normal
    │       ├── indiv
    │       └── joint
    │           └── Fig6
    │               ├── Fig6ABC
    │               └── fitResults
    │                   └── fitPlots
    └── src
```
FACS Data:
- SH1.119.xlsx (Figure 5)
- EX127.xlsx (Figure 6; data from Marchingo et al. Science 2014)
- EX130.xlsx (Repeat of EX127.xlsx)

Before fitting the cyton2 model to FACS datasets, please compile Cython file (`model.pyx`) in "code/cyton2/src" folder using custom Cython setup file, `_setup.py`. Typical command is `python _setup.py build_ext --inplace` in Mac/Linux terminal.