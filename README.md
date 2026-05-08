# UVLF Likelihood for Cobaya

Custom UV luminosity function (UVLF) likelihood module for the [Cobaya](https://github.com/CobayaSampler/cobaya) MCMC framework, based on the original implementation from the public GALLUMI likelihood(https://github.com/NNSSA/GALLUMI_public).

This likelihood supports:

* Combined HST + JWST UVLF datasets
* Tophat and sharp-k window function for halo mass function evaluation
* Asymmetric UVLF observational errors
* Redshift-dependent UVLF parameters
* Vectorized and cached computations for faster MCMC evaluation

---

## Authors

Sai Chaitanya Tadepalli
Siu Cheung Lam

Indiana University, Bloomington, USA

---

## Installation

Clone the repository:

```bash
git clone https://github.com/tsaic2808/UVLF_likelihood_cobaya.git
```

The likelihood can then be imported directly by Cobaya using:

```yaml
likelihood:
  uvlf:
    class: UVLF_model2.UVLF
    python_path: /path/to/UVLF_likelihood_cobaya
```

---

## Example Cobaya Usage

```yaml
likelihood:
  uvlf:
    class: UVLF_model2.UVLF
    python_path: .

    output_params:
      - chi2_hst
      - chi2_jwst

    CV: 0.2
    z_max: 12.5
    window: tophat
```

---

## Configurable Likelihood Options

| Option   | Description                                 |
| -------- | ------------------------------------------- |
| `CV`     | Cosmic variance floor applied to UVLF data  |
| `z_max`  | Maximum JWST redshift bin included          |
| `window` | Halo window function (`tophat` or `sharpk`) |
| `Anorm`  | ST normalization correction                 |
| `qnorm`  | ST `q` correction                           |
| `cnorm`  | Sharp-k cutoff parameter                    |

If `window: sharpk` is used, users should explicitly specify:

* `Anorm`
* `qnorm`
* `cnorm`

Example:

```yaml
likelihood:
  uvlf:
    class: UVLF_model2.UVLF
    python_path: .

    window: sharpk
    Anorm: 1.1
    qnorm: 1.2
    cnorm: 2.4
```

---

## Included Data

The repository includes:

* HST UVLF data (from z=4 to z=8)
* JWST UVLF data (from z=9 up to z=14.5)
* Dust-correction parameter tables

All file paths are internally portable using relative path handling.

---

## Citation

If using this likelihood, please cite:

* Original GALLUMI likelihood papers 2110.13161 and 2110.13168 
* Our UVLF cobaya based analysis paper 2512.16987

Additional citation information will be added here upon publication.

---

## Notes

This likelihood was adapted for Cobaya from the public GALLUMI implementation and redesigned for flexible MCMC analyses with CLASS/Cobaya pipelines.
