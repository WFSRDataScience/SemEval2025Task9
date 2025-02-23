from src.models.base_model import Model
from src.models.ml_model import MLModel
from src.models.transformers_model import TransformerModel


def get_model_name(model_name:str, config: dict) -> Model:
    if model_name in MLModel.NAMES:
        return MLModel(model_name, config)
    elif model_name in TransformerModel.NAMES:
        return TransformerModel(model_name, config)
    else:
        raise ValueError(f"Model name {model_name} not supported")