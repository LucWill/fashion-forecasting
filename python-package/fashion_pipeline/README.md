# Text Review Classification Pipeline

This package implements a modular machine learning pipeline for predicting product recommendation likelihood based on customer reviews. It includes components for data preprocessing, feature extraction, and model inference using scikit-learn and spaCy.

## Structure

### Modules

- `pipeline.py`  
  Defines the full feature engineering pipeline, including numerical, categorical, and textual transformations.

- `transformers.py`  
  Contains custom scikit-learn transformers, such as word counters or character analysis.

- `spacy_transformers.py`  
  Provides spaCy-based feature extraction (e.g., lemmatization, POS/NER features).

- `utils.py`  
  Includes helper functions for loading the trained model (`rf_model.pkl`) and review dataset (`reviews.csv`).

### Data

The `data/` folder contains:
- `rf_model.pkl`: Trained Random Forest model
- `reviews.csv`: Original dataset used for training and evaluation

## Usage

```python
from your_package import load_model, load_data, feature_engineering

model = load_model()
data = load_data()
X_transformed = feature_engineering.fit_transform(data)
predictions = model.predict(X_transformed)