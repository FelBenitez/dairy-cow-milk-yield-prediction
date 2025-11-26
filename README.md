# Dairy Cow Milk Yield Prediction 🐄📈

Predicting **daily milk yield (liters)** for dairy cows from rich farm, animal, and management data.

This repository contains the full pipeline we used for a **Kaggle-style course competition** – from data cleaning and feature engineering to **CatBoost / XGBoost modeling, Optuna tuning, and ensembling**.

> 🎯 **Result:** Final model achieved **4th place out of 52 teams (~192 students)** on the class Kaggle leaderboard.

---

## 🔍 Project Goals

- Build a **regression model** to predict `Milk_Yield_L` for each cow-day.
- Use **interpretable feature engineering** (farm behavior, feed, parity, time).
- Carefully manage **leakage**, **cross-validation**, and **generalization** to unseen farms.
- Explore **advanced modeling techniques**, including:
  - CatBoost as the primary learner  
  - XGBoost as a complementary model  
  - Optuna hyperparameter tuning  
  - Multi-seed and snapshot-style ensembling  
  - Stacking / blending using OOF predictions  

---

## 📦 Dataset

The dataset (provided by course staff) contains daily records for dairy cows, with:

### Target

- **`Milk_Yield_L`** – daily milk yield in liters (regression target).

### Animal-level features

- `Cattle_ID`, `Breed`, `Parity`
- `Previous_Week_Avg_Yield`

### Farm-level / management features

- `Farm_ID`
- Management / housing scores
- Feed type and quantity

### Time-related features

- `Date` (daily records over time)

Train/test CSVs live in `data/` and follow the usual Kaggle-style format:

- `train`: includes features + target  
- `test`: includes the same features but **no** `Milk_Yield_L`

---

## 🛠 Dependencies

Key libraries:

- `python`
- `pandas`, `numpy`
- `scikit-learn`
- `catboost`
- `xgboost`
- `optuna`

---

## 🧠 Modeling Overview

### Feature Engineering Highlights

- **Date features**
  - `year`, `month`, `day`, `dayofweek`, `weekofyear`, `quarter`, `date_ordinal`

- **Farm-level behavior**
  - **Fold-safe target encoding** of `Farm_ID` → `Farm_Performance`
  - Farm deltas:
    - `Prev_vs_Farm = Previous_Week_Avg_Yield - Farm_Performance`
    - `Prev_over_Farm = Previous_Week_Avg_Yield / (Farm_Performance + ε)`

- **Vaccination summary**
  - `Vax_Sum` = row-wise sum over all `*_Vaccine` flags

- **Parity indicators**
  - `First_Calf` (parity = 1)
  - `Prime_Cow` (parity 2–4)
  - `Old_Cow` (parity > 4)

- **Clustering (early experiments)**
  - We initially tried a `Farm_Cluster` feature via KMeans.
  - A first version **fit KMeans separately on train and test**, which made cluster labels inconsistent and hurt generalization.
  - After realizing this, we removed / reworked it and relied more on **farm target encoding + deltas**, which behaved much better in CV and on the leaderboard.

All preprocessing logic is encapsulated in helpers such as:

- `preprocess_pipeline_xgb`
- `fold_target_encode`
- `add_farm_deltas`

---

## 🎯 Hyperparameter Tuning

### CatBoost + Optuna

We ran **two Optuna studies** for CatBoost:

1. **Run 1 — 40 trials (core parameters)**  
   - Tuned: `depth`, `learning_rate`, `l2_leaf_reg`, `subsample`,  
     `random_strength`, `bagging_temperature`.  
   - `n_estimators` capped at 3000 with early stopping.  
   - **Best 5-fold CV RMSE:** ≈ **4.1064**  
   - This configuration became our **main CatBoost model**.

2. **Run 2 — 80 trials (expanded space)**  
   - Added: `border_count`, `min_data_in_leaf`, `bootstrap_type="Bayesian"`, etc.  
   - Tuned more tree-shape + regularization parameters.  
   - **Best 5-fold CV RMSE:** again ≈ **4.1064**.  
   - Confirmed we were already in a **flat optimum** – more complexity did not yield real gains.

### XGBoost + Optuna (500 trials)

To get a strong **complementary model**, we ran a **500-trial Optuna study** for XGBoost:

- 5-fold CV with **fold-safe preprocessing**:
  - farm target encoding,
  - farm deltas,
  - per-fold one-hot encoding + column alignment.
- Search space included:
  - `grow_policy ∈ {depthwise, lossguide}`
  - `max_depth`, `max_leaves`, `max_bin`
  - `learning_rate`, `subsample`, `colsample_*`
  - `gamma`, `reg_alpha`, `reg_lambda`

**Best trial:**

- Shallow but heavily regularized trees (`max_depth = 4`, `max_bin = 128`)
- Strong L1/L2 regularization and small learning rate
- **Best mean 5-fold CV RMSE:** ≈ **4.1151**

We then:

- Trained a **5-fold XGBoost ensemble** with early stopping.
- Built a **multi-seed full-data ensemble** using the tuned parameters.

XGBoost consistently performed **slightly worse** than CatBoost in RMSE, but its **error pattern was different**, making it useful for ensembling.

---

## 🤝 Ensembling & Stacking

We explored several ensemble strategies:

### 1. CatBoost Multi-Seed Ensemble

- Trained the **same tuned CatBoost configuration** with several random seeds.
- Averaged predictions across seeds.
- Effect: **small but consistent** improvement in CV and more stable leaderboard scores.

### 2. CatBoost + XGBoost Blending

- Generated OOF predictions for:
  - tuned CatBoost  
  - tuned XGBoost  
- Blended using  
  `y_blend = alpha * y_catboost + (1 - alpha) * y_xgboost`  
- Swept `alpha ∈ [0, 1]` and chose the value minimizing OOF RMSE.
- Used that same `alpha` to blend test predictions.
- Best blended OOF RMSE was **slightly better than pure XGBoost** and competitive with pure CatBoost.

### 3. Simple Stacking (Ridge Meta-Model)

- Built a small meta-dataset: `(CatBoost_OOF, XGBoost_OOF)`.
- Fit a **Ridge regression** meta-model to learn the optimal linear combo.
- In practice, gains were tiny and sometimes **less stable** on the leaderboard, likely due to overfitting OOF noise.

### 4. Snapshot-Style Ensembling (Hyperparameter Variants)

- Trained multiple CatBoost models with “nearby” hyperparameters:
  - varied `depth` (e.g., 5, 6, 7),
  - slightly scaled `learning_rate` around the tuned value.
- Combined this with multiple seeds per setting.
- Idea: average models living in **slightly different parts of parameter space** to reduce sensitivity and variance.

This snapshot ensemble was computationally expensive (multi-hour training) but conceptually similar to multi-seed ensembling across neighboring hyperparameter points.

---

## ✅ Cross-Validation & Robustness

To ensure we weren’t overfitting to a lucky split, we ran multiple CV checks:

- **Standard 5-fold KFold with shuffling**  
  - Default evaluation setup.

- **GroupKFold by `Farm_ID`**  
  - Entire farms held out in validation.  
  - GroupKFold RMSE was close to standard CV → suggests **reasonable farm-level generalization**.

- **Farm feature checks**
  - Tested models **with and without** farm-derived features (`Farm_Performance`, farm clusters, farm deltas).  
  - Removing all farm features **worsened RMSE**, indicating that the correctly implemented versions provide real signal.  
  - Also cleaned up a broken `Farm_Cluster` variant that trained KMeans separately on train and test (cluster label mismatch).

- **Time-related checks**
  - Compared models with/without trend-like date features (`year`, `date_ordinal`) to ensure we weren’t exploiting temporal artifacts.

These experiments increased our confidence that improvements were **real, not leakage-driven**.

---

## 🏁 Final Model & Performance

Our final submission is a **CatBoost-based ensemble** with:

- Best hyperparameters from **CatBoost Optuna Run 1**
- **Multi-seed ensembling**
- A careful **alpha blend** between:
  - CV-trained CatBoost ensemble  
  - full-data multi-seed CatBoost model  

- **Internal 5-fold CV RMSE:** ≈ **4.1061**  
- **Kaggle leaderboard placement:** **4th / 52 teams (~192 students)**  
