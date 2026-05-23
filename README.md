# Taylor Dispersion in Channels with Arbitrary Cross-Sectional Shapes

Code accompanying the paper "Taylor dispersion in long straight channels of arbitrary and gradually changing shapes".

## Installation

```bash
uv pip install -e .
```

For OpenFOAM data processing:
```bash
uv pip install -e ".[openfoam]"
```

For running notebooks:
```bash
uv pip install -e ".[notebook]"
```

## Project Structure

```
taylor_utils/
  io_utils.py            - Pickle save/load helpers
  moments.py             - Statistical moment functions (mean, variance, skewness, kurtosis)
  velocity.py            - Velocity field reconstruction for rectangular channels
  simulations.py         - Brownian dynamics simulations (rectangular, circular, dumbbell)
  rect_psi.py            - Analytical <u'Psi> for rectangular channels
  fdm_solver.py          - FDM Poisson solver for stream function in dumbbell channels
  inverse_problem.py     - Inverse problem: find channel shape for a target variance
  diffusion_timescale.py - Monte Carlo estimator for cross-sectional diffusion timescale
  openfoam.py            - OpenFOAM velocity field processing and interpolation

notebook/
  run_simulations_create_figs.ipynb - Main notebook: run simulations and generate figures
```

## Usage

```python
from taylor_utils import simulation_rectangular, simulation_circular, simulation_dumbell
from taylor_utils import create_velo_full, Psi_uPrime_avg
from taylor_utils import solve_inverse_problem, sigma2_const
from taylor_utils.openfoam import build_openFoam_interp
```

## OpenFOAM Cases

The `dumbell_geo_short_quarter_re*` directories contain OpenFOAM case setups for the dumbbell geometry at different Reynolds numbers. Large solution files are gitignored; only the case setup (`system/`, `constant/`, `0/`) is tracked.

## AI Use

Most of this code was developed by Caleb J. Samuel and Ray Chang. However, Claude Code was used to refactor and document the code. 

## Citation

If you use this code, please cite the accompanying paper.
