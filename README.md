# Surgery Duration Prediction

Machine learning project for predicting ENT surgery duration using historical clinical and surgical data from Hillel Yaffe Medical Center.

The goal was to improve operating-room scheduling by estimating surgery duration from information available before the procedure.

## Repository Contents

This public portfolio version includes:

- A public dataset derived from the hospital-approved project data
- The machine-learning training pipeline
- Project presentations
- Summary documentation
- Selected experiment notes and results

For additional privacy protection, direct record identifiers (`Patient`, `Medical Record`, and `Case Number`) were removed from the GitHub dataset because they are not needed for modeling or for demonstrating the project.

## Problem

Operating rooms are expensive, high-demand hospital resources. Inaccurate duration estimates can create scheduling gaps, delays, overtime and inefficient use of medical staff and facilities.

The project formulates this operational problem as a supervised regression task with **Total Surgery Time** as the prediction target.

## Data

The project evolved through several dataset versions during collection, cleaning and feature engineering. The public dataset contains thousands of surgical records and a broad set of clinical and operational features.

Feature groups include:

- Patient demographics
- BMI and vital signs
- Surgery and anesthesia characteristics
- Day-of-week information
- Medical conditions and allergies
- Pre-operative laboratory measurements

The public dataset is available at:

```text
data/public_surgery_dataset.csv
```

## Machine Learning Pipeline

1. Data cleaning and preprocessing
2. Missing-value handling
3. Categorical feature encoding
4. Outlier filtering using the IQR rule
5. Removal of highly correlated predictors
6. Train/test split
7. Feature scaling
8. Regression model training and comparison
9. Evaluation using Mean Absolute Error (MAE)

## Models Explored

The project included experiments with:

- Neural Networks
- XGBoost
- LightGBM
- Ensemble / Stacking models

The public training script implements a stacking ensemble combining LightGBM and XGBoost regressors.

## Evaluation

**Mean Absolute Error (MAE)** was used as the main metric because the prediction error can be interpreted directly in minutes.

Documented experiment milestones include:

| Model / Experiment | MAE |
| --- | ---: |
| Neural Network | 20.98 min |
| XGBoost | 22.22 min |
| Later ensemble experiment | ~18.77 min |

The project evolved across multiple dataset and preprocessing versions, so these values represent experiment milestones rather than a controlled benchmark under identical conditions.

## Project Structure

```text
surgery-duration-prediction/
├── data/
│   └── public_surgery_dataset.csv
├── src/
│   └── train_stacking_model.py
├── docs/
│   ├── mid-project-presentation.pptx
│   ├── project-presentation.pptx
│   ├── initial-data-summary.docx
│   ├── project-summary-part-a-redacted.pdf
│   └── final-project-summary-redacted.pdf
├── results/
│   └── experiment-notes.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Running the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate the stacking model:

```bash
python src/train_stacking_model.py data/public_surgery_dataset.csv
```

The script prints the MAE on the held-out test set and saves the fitted model bundle locally.

## Technologies

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `LightGBM` `Machine Learning` `Regression` `Feature Engineering` `Healthcare Data`

## What This Project Demonstrates

- Working with real-world healthcare data
- Translating an operational problem into a machine-learning task
- Data cleaning and feature engineering
- Regression model comparison
- Ensemble learning
- Evaluation with an interpretable metric
- Responsible preparation of data for a public technical portfolio
