import os
import pickle
from pathlib import Path


# Path to the root of the project
project_root = Path(__file__).parent

# Path to the trained model file
model_path = project_root / 'data' / 'gb_model.pkl' # / 'data' / 'mlp_100_50_model.pkl'


def load_model():
    """
    Load and return the trained machine learning model from disk.

    Priority:
    1. Use the path set in the MODEL_PATH environment variable (if provided)
    2. Fallback to the default model path relative to this script
    """

    # Allow override via environment variable
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        model_path = Path(model_path)
    else:
        model_path = Path(__file__).parent / "data" / "gb_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Set the correct path via the MODEL_PATH environment variable."
        )

    try:
        with model_path.open("rb") as file:
            return pickle.load(file)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Failed to load model due to missing module: {e}. "
            f"This may happen if the model was trained with Apple-only extensions like 'thinc_apple_ops'. "
            f"Consider re-saving the model without Apple-specific components."
        )