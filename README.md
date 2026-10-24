# Benzene (C6H6) Estimation from Low-Cost Air-Quality Sensors

Regression project for **Domain 4 – Environment & Sustainability**: estimate benzene concentration
`C6H6(GT)` (µg/m³) from metal-oxide sensor responses, temperature and relative humidity, with
chronological (no look-ahead) validation, a saved scikit-learn pipeline and a Streamlit + Plotly dashboard.

## Dataset findings (measured on `data/raw/training.csv`)
- 9,055 hourly rows × 15 columns, 2004-03-10 18:00 → 2005-04-04 14:00; **comma-delimited with dot decimals**
  (the problem statement says semicolon – the loader auto-detects both `,` and `;`+decimal comma).
- Missing readings are encoded as **−200** (not blank). 363 rows (4.0%) have every sensor and the target missing
  (instrument outages) and are dropped; no duplicate timestamps; no non-numeric values.
- `NMHC(GT)` is ~90% missing → excluded. Reference analysers (`CO/NMHC/NOx/NO2(GT)`) and derived `AH` are not used,
  because the goal is estimation from cheap sensors.
- **Important:** in this file `C6H6(GT)` is an almost deterministic function of `PT08.S2(NMHC)` (a quadratic in that
  single column reaches R² ≈ 0.9998). Scores that include S2 are therefore optimistic. `train.py` also cross-validates the
  selected model **without S2** as a sensitivity check, and the dashboard shows both.

## Method
Raw CSV → validation → cleaning (sentinel → NaN, bounds, de-duplication, time sort) → **chronological split**
(first 80% train / last 20% test) → `ColumnTransformer` (median impute + standardise, fitted on training folds only) →
5-fold expanding-window `TimeSeriesSplit` CV on the training period (Baseline mean, Linear, Ridge, Decision Tree,
Random Forest) → model chosen by CV **MAE** → final fit → **single** test evaluation (MAE, RMSE, R²) → joblib save.

## Run on Windows (PowerShell, from the project root)
```powershell
cd path\to\ml_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   (then re-run the line above)
# or use cmd.exe:            .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py                      # trains + saves models/best_pipeline.joblib and reports/
python -m streamlit run app.py       # dashboard
python -m pytest -v                  # tests
```
Other commands: `python train.py --data path\to\file.csv`, `python train.py --quick` (smoke test only),
`python predict.py --input readings.csv --output predictions.csv`.

## Files
| File | Responsibility |
|---|---|
| `config.py` | paths, target/features, split, CV, sentinel, bounds |
| `data_preprocessing.py` | loading (delimiter/encoding sniffing), validation, cleaning, chronological split, preprocessor |
| `pipeline.py` | candidate models, Pipeline factory, save/load with clear errors |
| `evaluation.py` | metrics, time-series CV, model selection, residuals, permutation importance |
| `eda.py` | PNG figures to `reports/figures/` |
| `train.py` / `predict.py` | training CLI / prediction API + CLI |
| `app.py` | dashboard (loads the saved model; never trains on open) |
| `tests/test_pipeline.py` | automated tests |

## Changing things
- **Target / features / split / selection metric:** edit `config.py` (`TARGET_COLUMN`, `FEATURE_COLUMNS`, `TEST_FRACTION`,
  `SELECTION_METRIC`, `ABLATION_FEATURES`), then run `python train.py`.
- **Models or hyper-parameters:** edit `get_model_candidates()` in `pipeline.py`.
- **Retrain on all data for deployment:** not done by default so reported scores stay honest.

## Limitations
One site, ~1 year; metal-oxide sensors drift and depend on season; test period is later in time but same site.
Inputs outside the training range or with missing values are flagged. Linear models can predict negative values.
The target/S2 near-determinism described above limits what the headline score means.
