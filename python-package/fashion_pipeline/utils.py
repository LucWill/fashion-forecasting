import pickle
from pathlib import Path


# Path to the root of the project
project_root = Path(__file__).parent

# Path to the trained model file
model_path = project_root / 'data' / 'gb_model.pkl' # / 'data' / 'mlp_100_50_model.pkl'


def load_model():
    """Load and return the machine learning model from disk."""
    with model_path.open('rb') as file:
        model = pickle.load(file)
    return model
