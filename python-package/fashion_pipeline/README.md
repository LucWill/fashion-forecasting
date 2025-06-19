# 🧵 fashion_pipeline – Text Review Classification Components

This package provides the core components of a modular machine learning pipeline designed to predict whether a customer will recommend a product based on their review. It includes text preprocessing, custom transformers, and model utilities using `scikit-learn` and `spaCy`.

## 📦 Module Overview

- **`pipeline.py`**  
  Defines the complete feature engineering pipeline:
  - Numerical features: scaling, imputation, log transformation
  - Categorical features: one-hot encoding
  - Text features: TF-IDF, lemmatization, POS/NER-based statistics

- **`transformers.py`**  
  Contains custom `scikit-learn` transformers such as:
  - Word and character counters
  - Digit detectors
  - Contrast word counters

- **`spacy_transformers.py`**  
  Implements advanced NLP features using `spaCy`, including:
  - Lemmatization
  - POS tag ratios
  - Named entity counts

- **`utils.py`**  
  Utility functions for:
  - Loading trained models (e.g., `gb_model.pkl`, `lr_model.pkl`)

- **`report/dashboard.py`**  
  Interactive prediction interface for text reviews using a `GradientBoostingClassifier`.

## 📁 Data Directory

The `data/` folder includes:
- `reviews.csv`: Cleaned dataset used for training and evaluation
- `gb_model.pkl`: Trained Gradient Boosting model
- `lr_model.pkl`: Trained Logistic Regression model

## 🧪 Example Usage

```python
from fashion_pipeline.utils import load_model, load_data

model = load_model("gb_model.pkl")
X_transformed = feature_engineering.fit_transform(data)
predictions = model.predict(X_transformed)
```

You can also load a dictionary-style review and get a prediction:

```python
sample_review = {
    "Title": "Highly recommended",
    "Review Text": "Beautiful material, great fit!",
    "Age": 45,
    "Division Name": "General",
    "Department Name": "Dresses",
    "Class Name": "Dresses"
}

model.predict([sample_review])
```
