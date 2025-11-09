# 🏠 House Price Prediction (Kaggle – Ames Housing)

This repository contains an end-to-end **house price prediction** project based on the Kaggle competition  
**“House Prices: Advanced Regression Techniques”**. It demonstrates a full machine learning workflow:
data understanding, preprocessing, feature engineering, model training, evaluation, and prediction export.

---

## 📊 Project Overview

The goal is to predict the **SalePrice** of residential homes in Ames, Iowa using 79 explanatory variables  
(e.g., lot size, neighborhood, overall quality, year built, garage size).

Key steps in the pipeline:

1. Exploratory Data Analysis (EDA) in a Jupyter notebook
2. Data preprocessing (missing values, encoding categoricals, scaling numeric features)
3. Model training with regularized regression models
4. Model evaluation using RMSE and R²
5. Prediction generation on the Kaggle test set in submission format

---

## 🧠 Models

The scripts and notebook are set up to support:

- Ridge Regression (RidgeCV)
- Lasso / ElasticNet
- (Optional) Stacking / Blending models

You can easily extend the project with tree-based models (RandomForest, XGBoost, LightGBM, etc.).

---

## 🧩 Tech Stack

- Language: Python 3.9+
- Core libraries:
  - pandas, numpy – data handling
  - scikit-learn – preprocessing, models, metrics, pipelines
  - matplotlib, seaborn – visualization
  - joblib – model persistence
  - jupyter – notebooks

---

## 📂 Project Structure

```bash
house-price-prediction/
├── data/
│   ├── train.csv                # Kaggle training data (NOT committed by default)
│   ├── test.csv                 # Kaggle test data   (NOT committed by default)
│   ├── sample_submission.csv    # Sample submission format
│   └── data_description.txt     # Description of all features
│
├── notebooks/
│   └── house_prices_blend_stack.ipynb   # Main analysis & experimentation notebook
│
├── src/
│   ├── preprocess.py            # Functions for loading & preprocessing data
│   ├── train_model.py           # Script to train and save models
│   └── evaluate_model.py        # Script to evaluate trained models
│
├── models/
│   └── (saved models go here, e.g. ridge_model.pkl)
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

Note: The raw Kaggle CSV files can be large, so they are excluded via `.gitignore`.  
To run the project, download the dataset from Kaggle and place the files in the `data/` folder.

---

## 🚀 How to Run Locally

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/house-price-prediction.git
   cd house-price-prediction
   ```

2. Create and activate a virtual environment (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Add data files

   Download the Kaggle dataset "House Prices: Advanced Regression Techniques" and place:

   - train.csv
   - test.csv
   - sample_submission.csv
   - data_description.txt

   into the `data/` folder.

5. Run the notebook (EDA + experiments)

   ```bash
   jupyter notebook notebooks/house_prices_blend_stack.ipynb
   ```

6. Train a model from the command line

   ```bash
   python -m src.train_model
   ```

   This will:
   - Load `data/train.csv`
   - Preprocess features
   - Train a regularized regression model (Ridge by default)
   - Save the trained model to `models/ridge_model.pkl`

7. Evaluate the model

   ```bash
   python -m src.evaluate_model
   ```

   This will:
   - Load the trained model and training data
   - Produce evaluation metrics (RMSE, R²) on a validation split

---

## 🧪 Configuration

Default settings (model type, validation split size, random seed, etc.) are defined inside
`src/train_model.py` and `src/evaluate_model.py`. You can modify these scripts to:

- Switch between Ridge, Lasso, ElasticNet
- Change cross-validation strategy
- Add new models (e.g., Gradient Boosting, XGBoost, LightGBM)
- Adjust feature preprocessing logic

---

## 📚 Dataset

The dataset describes various characteristics of residential homes in Ames, Iowa.  
Examples of features (see `data/data_description.txt` for full detail):

- MSSubClass – Type of dwelling (1-Story, 2-Story, PUD, etc.)
- MSZoning – Zoning classification
- LotArea – Lot size in square feet
- Neighborhood – Physical location within Ames
- OverallQual – Overall material and finish quality
- YearBuilt – Original construction date
- GrLivArea – Above-grade (ground) living area in square feet
- GarageCars – Garage size in car capacity
- SaleCondition – Condition of sale

Target variable:
- SalePrice – Final price of the house in USD.

---

## 👩‍💻 Author

**Renu Singh Parihar**  
Tampa, Florida, USA  

- LinkedIn: https://www.linkedin.com/in/renu-singh-parihar
- GitHub:  https://github.com/<your-username>

Feel free to open issues or suggestions if you would like to extend this project with more models,
hyperparameter tuning, or deployment (for example, Streamlit or FastAPI).
