import os
import pickle
from pathlib import Path


# Path to the root of the project
project_root = Path(__file__).parent

# Path to the trained model file
model_path = project_root / 'data' / 'gb_model.pkl'


def load_model(model_filename: str = "gb_model.pkl"):
    """
    Load and return the trained machine learning model from disk.

    Parameters
    ----------
    model_filename : str, optional
        Name of the model file to load from the `data/` folder
        (default: 'gb_model.pkl').

    Priority:
    1. Use the full path set in the MODEL_PATH environment variable
        (if provided)
    2. Fallback to the specified model filename in
        the default `data/` folder
    """

    # Check if the environment variable overrides the model path
    env_model_path = os.getenv("MODEL_PATH")
    if env_model_path:
        model_path = Path(env_model_path)
    else:
        model_path = Path(__file__).parent / "data" / model_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Set the correct path via the MODEL_PATH environment variable"
            f"or check the filename."
        )

    try:
        with model_path.open("rb") as file:
            return pickle.load(file)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Failed to load model due to missing module: {e}. "
            f"This may happen if the model was trained with Apple-only "
            f"extensions like 'thinc_apple_ops'. Consider re-saving the "
            f"model without Apple-specific components."
        )
