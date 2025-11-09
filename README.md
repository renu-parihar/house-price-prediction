# House Prices Pridiction — Project

This repository is a ready-to-run project skeleton for working with the Ames House Prices dataset. It includes the data description you provided plus helper scripts to train a simple baseline model (Random Forest) and produce a submission file.

What you get:
- Project folders (data, src, models, outputs)
- data_description.txt (your file)
- sample_submission.csv (your file)
- test.csv (placeholder — replace with your full test.csv if not included)
- Simple training & prediction scripts that train if you provide train.csv, otherwise will copy sample_submission.csv as fallback
- Requirements and run instructions

Quick start
1. Install dependencies:
   pip install -r requirements.txt

2. Place your full `train.csv` (Kaggle train dataset) into the `data/` folder if you want to train a model:
   data/train.csv

3. To run the full pipeline (train -> predict):
   bash run.sh

   - If `data/train.csv` exists the script will train a RandomForest baseline and save a model in `models/` and generate `outputs/submission.csv`.
   - If `train.csv` is not present the script will copy `data/sample_submission.csv` to `outputs/submission.csv` as a placeholder.

Project layout
- data/
  - data_description.txt
  - test.csv
  - sample_submission.csv
  - (optional) train.csv  ← place here if you want to train
- src/
  - train.py       ← train baseline model
  - predict.py     ← generate submission from test.csv and saved model
  - utils.py       ← helper functions
- models/
  - (generated) baseline_model.joblib
- outputs/
  - (generated) submission.csv
- run.sh
- requirements.txt

Notes
- The baseline uses only simple numeric features (no heavy preprocessing). It's a starting point — adapt feature engineering/modeling as needed.
- If you want the exact full test.csv included, paste it into `data/test.csv`. The provided `data/test.csv` file in this repo is a placeholder header + first rows for layout; replace it with your full file if necessary.

If you'd like, I can:
- include a full test.csv (as you shared) inside data/ (I can add it if you confirm),
- provide a Jupyter notebook with EDA and feature engineering,
- replace the baseline with a LightGBM pipeline.
