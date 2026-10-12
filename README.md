# MODALII LENSHUB COD return risk

An IT3091 Machine Learning group project investigating order-level return risk for MODALII LENSHUB (PVT) LTD. The intended use is to prioritise manual confirmation before dispatch.

The study compares a majority baseline with logistic regression, a decision tree, random forests, histogram gradient boosting and XGBoost. Model selection uses five-fold cross-validation inside a fixed stratified training split. Threshold selection uses training out-of-fold F1 before holdout evaluation.

## Results

The training-selected model is a random forest with SMOTENC. On the 864-order retrospective holdout:

| Metric | Value |
| --- | ---: |
| Average precision | 0.245 |
| ROC-AUC | 0.585 |
| Returned-class precision | 22.48% |
| Returned-class recall | 24.37% |
| Returned-class F1 | 0.234 |

At the frozen threshold of 0.5164, it identifies 29 returns, flags 100 successful orders and misses 90 returns. The evidence supports only a monitored confirmation pilot. It does not support automatic rejection or a claim of financial savings.

## Repository structure

```text
src/          Shared workflow and command-line entry point
notebooks/    Complete study and four role notebooks
data/         Data dictionary and dataset instructions
results/      Aggregate comparison tables and figures
docs/         Decision logs, preprocessing notes and provenance
reports/      Group report in PDF and editable Word formats
```

## Run locally

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-notebooks.txt
python src/run_project.py --dataset "C:\path\to\Modalii_Lenshub_ML_Dataset_Final.csv"
```

On macOS or Linux, activate with `source .venv/bin/activate`.

Alternatively, place the approved CSV at `data/Modalii_Lenshub_ML_Dataset_Final.csv` and run `python src/run_project.py`. Results are regenerated under `results/`. The private dataset and row-level outputs are ignored by Git.

## Notebooks and Colab

Open `notebooks/00_complete_study.ipynb` and run its cells in order. Local notebook execution expects the CSV in `data/`. In Colab, the setup cell installs pinned dependencies and prompts for the approved dataset CSV. Notebook outputs and execution metadata are cleared for a clean repository; the executed submission copies are retained separately.

## Data and limitations

The client dataset is not distributed in this repository. See [dataset instructions](data/README.md) and [data provenance](docs/data_provenance.md). The reported target construction uses a Waybill Number match between delivery and return reports. Original exports and merge code were unavailable for independent verification.

All 206 zero-COD records are retained and flagged; their operational meaning is unconfirmed. No dates or customer groups are available, and this is retrospective validation on a previously explored dataset. Scores are not calibrated probabilities.

## Report and submission

The report contains the full analysis and references the separately organised assignment submission package. Add actual stakeholder feedback and the final YouTube link before submitting it. Each group member writes their own one-A4-page personal learning report.

The assignment's required [AI-use declaration](docs/AI_USE.md) is retained. The group remains responsible for reviewing, understanding and defending the submitted work.
