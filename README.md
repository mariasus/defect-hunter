# Defect analysis scripts for irradiated Fe70Cr15Ni15

OVITO Python scripts to identify and quantify radiation-induced defects in molecular dynamics simulations of an Fe70Cr15Ni15 austenitic solid solution.

These scripts accompany the article:

> M. Susini, L. Van Brutzel, A. Chartier, S. Argirò, D. Tomatis,
> *Interplay of dose and temperature in defect evolution and microstructure development in irradiated Fe70Cr15Ni15 austenitic solid solution*,
> submitted to Acta Materialia (2026).

The methods, parameter choices and their limitations are described in the Supplementary Material of the article.

## Scripts

| Script | Identifies | Environment | Suppl. Mat. |
|---|---|---|---|
| `dumbbells.py` | Split interstitials | OVITO 3.1.3 | Sec. S2 |
| `void.py` | Vacancies and vacancy clusters | OVITO 3.12 | Sec. S3 |
| `a15.py` | A15 Frank-Kasper phase  | OVITO 3.1.3 | Sec. S4 |
| `sft.py` | Stacking Fault Tetrahedra | OVITO 3.1.3 | Sec. S5 |
| `stacks.py` | Staking faults, dissociated Perfect dislocations, dislocations | OVITO 3.1.3 | Sec. S6 |


## Requirements

- [OVITO 3.1.3](https://www.ovito.org), with its `ovitos` interpreter, for all scripts except the vacancy analysis
- Python 3 with the OVITO 3.12 module, for the vacancy analysis:

  ```bash
  pip install "ovito==3.12.*"
  ```

## Usage

Each script takes one or more atomic configurations as arguments:

```bash
ovitos script1.py config1.cfg.gz [config2.cfg.gz ...]
```

The vacancy analysis runs with standard Python:

```bash
python vacancy_script.py config1.cfg.gz [config2.cfg.gz ...]
```

Configurations should be energy-minimised before the analysis (we used conjugate gradient minimisation in LAMMPS).

## Important

The analysis parameters are calibrated for Fe70Cr15Ni15 described with the Béland et al. interatomic potential. Recalibrate them before applying the scripts to other compositions or potentials.

## Citation

If you use these scripts, please cite the article above (DOI to be added) and this repository (Zenodo DOI to be added).

## License

Released under the MIT License. See `LICENSE`.
