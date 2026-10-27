# Notebook

Follow the workflow below with the project Python kernel. All modeling, training, validation and recommendation operations call `src/hlo_recommender`.

| Notebook | Operation |
|---|---|
| [Setup](Setup/setup.ipynb) | Clone, install, inspect and verify |
| [Data](Data/data.ipynb) | Load, validate, split and serialize |
| [mT5](Training/mt5.ipynb) | Preflight, download and fine-tune |
| [BKT](Training/bkt.ipynb) | Fit and update mastery |
| [Policy](Training/policy.ipynb) | Optional transition-based training |
| [SPARQL](Queries/sparql.ipynb) | Ground, generate, validate and execute |
| [Evaluation](Evaluation/evaluation.ipynb) | Query, prediction and ranking metrics |
| [Recommendation](Recommendation/recommendation.ipynb) | Retrieve, adapt and export |

```bash
git clone https://github.com/asgmodel/recommendation-system.git
cd recommendation-system
python -m pip install -e . jupyterlab ipykernel
python -m jupyter lab Notebook
```

Development records and relevance grades are synthetic. Existing data is loaded directly; no external collection or chatbot-generated dataset is claimed.

Enable `DOWNLOAD_BASE` and `TRAIN_MODEL` in `Training/mt5.ipynb` for neural training; enable `RUN_MT5` in `Evaluation/evaluation.ipynb` and set `GENERATOR = "mt5"` in `Queries/sparql.ipynb` and `Recommendation/recommendation.ipynb` afterward.

Policy training is optional and requires reviewed transitions. Outputs and notebook session paths are stored under `artifacts/notebooks/`; training notebooks create a new run directory each time.
