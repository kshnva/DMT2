# Hotel Booking Prediction and Ranking

Learning-to-rank models for hotel search result ranking (Expedia-style), completed as part of the Data Mining Techniques course at VU Amsterdam.

## Overview

Given user search queries and hotel properties, the task is to rank hotels by booking likelihood. The approach combines gradient-boosted ranking models with a meta-learning ensemble.

## Approach

1. **Exploratory Data Analysis** (`eda.ipynb`) — distribution analysis, missing value assessment, target variable exploration
2. **Feature Engineering** (`feature_engineering.ipynb`) — property-level statistics, user–property interaction features, price normalization
3. **Model Training** (`model.ipynb`, `Final_model.ipynb`) — training and evaluation of individual rankers and the stacked ensemble
4. **Configuration** (`configuration.py`) — centralized paths, hyperparameters, and model settings

## Models

- **XGBoost Ranker** — pairwise learning-to-rank with `rank:pairwise` objective
- **LightGBM Ranker** — lambdarank objective with NDCG optimization
- **Ridge Meta-Model** — stacks predictions from both rankers for final scoring

## How to Run

1. Place training and test CSV files in `datasets/`
2. Run notebooks in order: `eda.ipynb` → `feature_engineering.ipynb` → `Final_model.ipynb`
3. Submission files are written to `datasets/submissions/`

## Requirements

- Python 3.8+
- XGBoost, LightGBM
- scikit-learn
- pandas, NumPy, Matplotlib
