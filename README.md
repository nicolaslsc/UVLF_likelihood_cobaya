# UVLF Likelihood for Cobaya

Custom UV luminosity function (UVLF) likelihood module for the [Cobaya](https://github.com/CobayaSampler/cobaya) MCMC framework, based on the original implementation from the public GALLUMI likelihood.

This likelihood supports:

* Combined HST + JWST UVLF datasets
* Sheth–Tormen and sharp-k halo mass functions
* Asymmetric observational errors
* Alcock–Paczynski corrections
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

* HST UVLF data
* JWST UVLF data
* Dust-correction parameter tables

All file paths are internally portable using relative path handling.

---

## Citation

If using this likelihood, please cite:

* Original GALLUMI likelihood paper
* Associated UVLF analysis paper(s)

Additional citation information will be added here upon publication.

---

## Notes

This likelihood was adapted for Cobaya from the public GALLUMI implementation and redesigned for flexible MCMC analyses with CLASS/Cobaya pipelines.
