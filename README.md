
# 🧵 Fashion Forward Forecasting – StyleSense Recommendation Predictor

## 📌 Project Overview

**StyleSense** is a rapidly growing online retailer in women's fashion. As the customer base grows, the company faces a backlog of product reviews with missing information about whether a product was recommended. This project addresses that challenge by creating a **machine learning pipeline** to predict product recommendations based on review text, customer data, and product information.

The model analyzes:
- 📝 Review text (`Title` and `Review Text`)
- 🎂 Customer age
- 🧵 Product metadata (e.g., class, department)
- 📈 Feedback metrics

The output is a prediction: **Will the customer recommend the product or not?**

---

## 🧪 Project Structure

| File/Folder | Description |
|-------------|-------------|
| `notebook.ipynb` | Main notebook with data exploration, pipeline definition, model training & evaluation |
| `pipeline.py` | (Optional) Python module defining reusable pipeline components |
| `spacy_transformers.py` | (Optional) Custom NLP transformers (e.g., lemmatizer, POS/NER) |
| `model.pkl` | Trained model pipeline saved for deployment/inference |
| `README.md` | Project overview and instructions (this file) |
| `data/` | (Optional) Folder containing the raw dataset |

---

## 🧰 Technologies Used

- Python 3.x
- `pandas`, `numpy` for data handling
- `scikit-learn` for preprocessing, modeling, and pipeline structure
- `spaCy` for NLP (lemmatization, POS/NER)
- `matplotlib`, `seaborn` for visualization
- (Optional) `Streamlit` or `Gradio` for interactive dashboard

---

## ⚙️ Pipeline Overview

The ML pipeline handles **preprocessing, feature engineering, training, and inference** in one structure:

### Preprocessing Steps:
- Imputation of missing values (numeric & categorical)
- Scaling for numerical features
- One-hot encoding for categorical features
- TF-IDF vectorization for text fields
- Custom NLP features (e.g., lemmatization, POS/NER)

### Model:
- `RandomForestClassifier` (initial baseline)
- Tuned using `GridSearchCV` for better performance

### Evaluation:
- Train/test split
- Classification metrics: accuracy, precision, recall, F1-score
- Visualization of results and feature importance

---

## 📊 Results

- Achieved **X% accuracy** on the test set
- Insights:
  - Text features provided strong predictive power
  - Customers in certain product categories were more likely to recommend items
  - Feedback counts and review length also showed signal

*(Update with actual performance numbers and insights.)*

---

## 🧠 Future Improvements

- Incorporate word embeddings or transformer-based NLP (e.g., `spaCy` vectors, BERT)
- Explore sentiment analysis
- Build an interactive prediction dashboard
- Deploy pipeline with a front-end using Streamlit or Gradio

---



## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Running the Notebook
```bash
jupyter notebook notebook.ipynb
```

### Using the Trained Model
```python
import joblib
pipeline = joblib.load('model.pkl')
prediction = pipeline.predict([new_data_dict])
```

---

### Dependencies

```
Examples here
```

### Installation

Step by step explanation of how to get a dev environment running.

List out the steps

```
Give an example here
```

## Testing

Explain the steps needed to run any automated tests

### Break Down Tests

Explain what each test does and why

```
Examples here
```

## Project Instructions

This section should contain all the student deliverables for this project.

## Built With

* [Item1](www.item1.com) - Description of item
* [Item2](www.item2.com) - Description of item
* [Item3](www.item3.com) - Description of item

Include all items used to build project.

## License

[License](LICENSE.txt)

## 🧑‍💻 Author

Luc Will  
M.Sc. Physicist, aspiring Data Scientist  
(You can include LinkedIn, GitHub links, or Udacity info here.)
