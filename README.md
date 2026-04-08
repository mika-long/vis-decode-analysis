# Overview 

For **section 4**, the compiled html files for each decoding operator: 
- Section ..., 
- Section ..., 

For **section 5**, the compiled html file correspondg to each section: 
- Section ... -> 


## Repository structure 

- 📁 `analysis\`: contains Quarto notebooks for analyzing results for each task 
  - `quarto_files\`: R files using `brms` and `ggplot2` to analyze results. 
    - `part1`: files related to Section 4 of the paper. 
    - `part2`: files related to Section 5 of the paper.  
  - 📁 `fitted_models`: contains fitted `brms` models, available on [OSF](https://osf.io/prtfq/files/osfstorage?view_only=1c7104a9488940a6aa5a042d41bb1232).  
  - `stan_files`: the `Stan` makefiles used by `cmdstanr` 
  - `stan_bin_files`: compiled stan binaries; not tracked 
- 📁 `data\`: data collected from experiments 
  - `moritz`: stimuli and experiment data from Moritz et al. 
  - `part1`: experiment data collected from the first experiment, used to quantify operators 
  - `part2`: experiment data collected from the second experiment, used to demonstrate its ability to generalize 
- 📁 `output\`: where compiled `.qmd` files, in the form of `html`, sits. 
- 📁 `R`: folder for R scripts 
  - `helper.R`: helper functions used across all tasks 
- 📁 `figs`: contains rendered figures used in the preprint. 