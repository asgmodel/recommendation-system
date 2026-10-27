# Notebook

Run the notebooks in numeric order with the project Python kernel. All modeling, training, validation and recommendation operations call `src/hlo_recommender`.

| Order | Notebook | Operation |
|---|---|---|
| 00 | [Setup](00_Setup/00_setup.ipynb) | Clone, install, inspect and verify |
| 01 | [Data](01_Data/01_data.ipynb) | Load, validate, split and serialize |
| 02 | [mT5](02_Training/02_mt5.ipynb) | Preflight, download and fine-tune |
| 03 | [BKT](02_Training/03_bkt.ipynb) | Fit and update mastery |
| 04 | [Policy](02_Training/04_policy.ipynb) | Optional transition-based training |
| 05 | [SPARQL](03_Queries/05_sparql.ipynb) | Ground, generate, validate and execute |
| 06 | [Evaluation](04_Evaluation/06_evaluation.ipynb) | Query, prediction and ranking metrics |
| 07 | [Recommendation](05_Recommendation/07_recommendation.ipynb) | Retrieve, adapt and export |

```bash
git clone https://github.com/asgmodel/recommendation-system.git
cd recommendation-system
python -m pip install -e . jupyterlab ipykernel
python -m jupyter lab Notebook
```

Development records and relevance grades are synthetic. Existing data is loaded directly; no external collection or chatbot-generated dataset is claimed.

Enable `DOWNLOAD_BASE` and `TRAIN_MODEL` in notebook 02 for neural training; enable `RUN_MT5` in notebook 06 and set `GENERATOR = "mt5"` in notebooks 05 and 07 afterward.

Policy training is optional and requires reviewed transitions. Outputs and notebook session paths are stored under `artifacts/notebooks/`; training notebooks create a new run directory each time.
