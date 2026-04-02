## Repository structure 

- 📁 `analysis\`: contains Quarto notebooks for analyzing results for each task 
  - `pymc_files\`: python files using `PyMC` and `arivz` to analyze results 
  - `quarto_files\`: R files using `brms` and `ggplot2` to analyze results 
  - 📁 `fitted_models`: contains fitted `brms` models as well as `.nc` models, available on [OSF](https://osf.io/prtfq/files/osfstorage?view_only=1c7104a9488940a6aa5a042d41bb1232).  
  - `stan_files`: the `Stan` makefiles, shared by both workflows (R uses `cmdstanr` while python uses `cmdstanpy`)
  - `stan_bin_files`: compiled stan binaries; not tracked 
- 📁 `data\`: data collected from experiment, both raw and processed 
- 📁 `output\`: where compiled `.qmd` files, in the form of `html`, sits. 
- 📁 `R`: folder for R scripts 
  - `helper.R`: helper functions used across all tasks 
 
- 📁 `figs`: contains rendered figures used in the preprint. 