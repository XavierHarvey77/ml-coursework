# Regression Model Comparison on Letterboxd Movie Metadata

This repository contains a machine learning coursework project analyzing structured movie metadata from Letterboxd. The project compares multiple regression models to evaluate how well film metadata can predict a constructed engagement-related target, with emphasis on preprocessing, feature engineering, model comparison, and empirical evaluation.

## Project Overview

The goal of this project was to practice an end-to-end machine learning workflow using a real-world tabular dataset. The dataset included movie-level metadata such as release year, runtime, title length, description word count, and other derived numerical features. After cleaning and preprocessing the data, I trained and compared several regression models to understand which modeling approach performed best and why.

The project was developed for an AI Fundamentals course and reflects my interest in empirical machine learning, model evaluation, and performance tradeoffs.

## Motivation

As both a film enthusiast and a Mathematics & Computer Science graduate, I wanted to work with a dataset connected to cinema while applying machine learning methods in a rigorous way. Letterboxd provided a useful domain because film metadata contains interpretable features that can be used to explore questions about prediction, recency, popularity, and user engagement.

The central research question was:
Can structured movie metadata be used to predict whether a film is likely to have recent user engagement?

## Dataset

The project used a Letterboxd movie metadata dataset sourced from Kaggle. The dataset included information such as:
- Movie title
- Release year
- Runtime
- Description/logline text
- Recency-related indicators
- Derived numerical features

The original dataset required preprocessing before modeling, including handling missing numerical values and constructing features suitable for regression.

## Features Used

The final model used six numerical features derived from the dataset:
- **year**: release year of the film
- **age_years**: age of the movie based on release year
- **runtime**: movie duration in minutes
- **title_length**: number of characters in the film title
- **word_count**: number of words in the movie description
- **title_runtime_ratio**: ratio of title length to runtime

Missing numerical values were handled using median imputation with **SimpleImputer** from **sklearn.impute.**

## Target Variable

The dataset did not directly include a continuous user-engagement score, so I constructed a numerical prediction target from the available **is_recent** field:

```
log_likelihood = is_recent.astype(int)
```

log_likelihood = is_recent.astype(int)

This target was used to approximate whether a film had recent logging activity on Letterboxd. Although the target is binary in origin, the project was framed as a regression task for the purpose of comparing regression models and prediction error.

## Models Compared

Three regression approaches were implemented and compared:

### Lasso Regression
Lasso Regression was used as a simple, interpretable linear baseline. Because it applies L1 regularization, it can shrink less useful feature coefficients toward zero, making it useful for feature selection and baseline comparison.

### Random Forest Regression
Random Forest Regression was used to capture nonlinear relationships in the metadata. Since it aggregates predictions across many decision trees, it can model more complex patterns than linear regression while also providing feature-importance information.

### Polynomial Regression
Polynomial Regression was used to test whether nonlinear feature transformations could improve performance over a purely linear baseline. Polynomial features were generated with degree 3 to increase expressiveness while avoiding excessive complexity.

## Train/Test Split

The dataset was divided using an 80/20 train-test split:

```
train_test_split(X, y, test_size=0.2, random_state=42)
```

The 80/20 split provided enough training data for model learning while reserving a separate test set for evaluation. The fixed random seed ensured reproducibility.

## Evaluation Metrics

Models were evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

These metrics were used to compare prediction error across the three model types.

## Results

Random Forest Regression performed best among the tested models, achieving the lowest RMSE. This suggested that the relationship between movie metadata and recent logging behavior was nonlinear and better captured by an ensemble method than by purely linear approaches.

The overall results were:
- Random Forest Regression performed best.
- Polynomial Regression improved over Lasso Regression.
- Lasso Regression performed worst, likely because the metadata-target relationship was not primarily linear.
- Movie recency was the strongest predictor.
- Runtime alone was not highly predictive, but it became more useful when combined with other features.
- Text-derived features such as title length and description word count provided modest predictive value.

## Key Learning Outcomes

This project strengthened my understanding of:
- Data preprocessing
- Median imputation for missing values
- Feature engineering
- Train/test splitting
- Regression modeling
- Model comparison
- MSE and RMSE evaluation
- Nonlinear model behavior
- Feature importance
- The importance of preventing data leakage
- The value of reproducible machine learning workflows

It also helped me think more carefully about how different model families can approach the same prediction task in different ways, and how empirical evaluation can reveal tradeoffs between interpretability, flexibility, and predictive performance.

