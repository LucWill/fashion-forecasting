# 📝 Text Review Classification Pipeline

**Predict customer product recommendations from text reviews using NLP and machine learning.**  
This project showcases end-to-end ML pipeline design, from preprocessing and feature engineering to model evaluation and deployment, with a focus on **interpretability and modularity**.

## 💼 Project Summary

- **Goal**: Predict whether a customer would recommend a product based on their written review and metadata.
- **Accuracy**: Achieved **91% accuracy** on the test set using an `MLPClassifier`.
- **NLP**: Used `spaCy` for lemmatization, POS/NER, and custom transformers.
- **ML Models**: Compared `RandomForest`, `GradientBoosting`, `MLP`, and `LogisticRegression`.
- **Deployment**: Built an **interactive prediction dashboard** for real-time text classification.
- **Code Quality**: Modular, test-driven design (`pytest`), reusable components, well-documented.

---

## 🧠 What This Project Demonstrates

✅ **Real-world ML pipeline** from raw text to prediction  
✅ **Explainable preprocessing** and custom NLP features  
✅ **Evaluation-driven modeling decisions**  
✅ **Software engineering skills**: clean package structure, automated tests, reproducibility

---

## 🗂️ Repository Structure

```
.
├── python-package/
│   └── fashion_pipeline/
│       ├── pipeline.py              ← Full ML pipeline (numerical, categorical, text)
│       ├── transformers.py          ← Custom sklearn transformers
│       ├── spacy_transformers.py    ← spaCy-based NLP feature extraction
│       ├── utils.py                 ← Load models and data
│       ├── data/
│       │   ├── reviews.csv
│       │   ├── gb_model.pkl
│       │   ├── lr_model.pkl
│       ├── report/dashboard.py      ← Interactive prediction interface
├── starter/starter.ipynb            ← Notebook for EDA, modeling, and insights
├── tests/                           ← Unit tests for each pipeline component
```

---

## 📊 Evaluation Highlights

- **Best model**: `MLPClassifier` with **91% test accuracy**
- **Textual features** provided the most predictive power
- **Numerical & categorical data** (e.g., age, department) had limited impact
- **Length of review & feedback count** didn’t significantly improve performance
- **Confusion matrix** used to evaluate precision/recall tradeoffs

---

## 🚀 Deployment & Usage

### Interactive Dashboard  
A custom lightweight HTML app allows users to input a review and receive instant prediction results from the `GradientBoostingClassifier`.

### Example API Use

```python
from fashion_pipeline.utils import load_model

# Load default model
model = load_model()

# Load alternative model from data/ folder
model = load_model("rf_model.pkl")

sample = {
    "Title": "Love this dress!",
    "Review Text": "The fit is amazing and the color is vibrant.",
    "Age": 28,
    "Division Name": "General",
    "Department Name": "Dresses",
    "Class Name": "Dresses"
}

model.predict([sample])  # Output: [1] (recommended)
```

---

## 🧪 Testing

Run all tests via:

```bash
pytest
```

### Test Coverage

- `test_pipeline.py`: Ensures preprocessing consistency  
- `test_transformers.py`: Validates custom features (e.g., contrast words, character counts)  
- `test_spacy_transformers.py`: Verifies NLP-based feature extraction  
- `test_dashboard.py`: Confirms correct dashboard functionality

---

## 📥 Installation

```bash
# 1. Clone repo
git clone https://github.com/LucWill/fashion_pipeline.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install package
pip install -e fashion_pipeline
```

## 📓 Explore the Project

```bash
jupyter notebook starter/starter.ipynb
```

---

## 🛠️ Built With

- **Python 3.10**
- **scikit-learn** – ML modeling and pipelines  
- **spaCy** – NLP preprocessing and POS/NER  
- **pandas / numpy** – Data analysis and manipulation  
- **pytest** – Test automation  
- **fasthtml** – Custom web dashboard (no Streamlit/Gradio)

---

## 📄 License

Distributed under the [MIT License](LICENSE.txt).
