import pandas as pd
import argparse
import shap
import pickle

import src.common.tools as tools
import src.data.dataio as dataio
from src import logging
from src.models import get_model_name, TransformerModel

logger = logging.getLogger(__name__)

pd.options.display.max_colwidth = 300

def predict(config: dict, category: str, test_df: pd.DataFrame, index: int) -> None:
    modelname = config['modelname']
    logger.info(f"Predicting model {modelname} for {category} augmenter {config['augmenter_name']}")

    model = get_model_name(modelname, config)
    model.set_params(config, config["field"], category)
    model.initialize_model_predict()

    row = test_df.iloc[index]
    logger.debug(row["text"])
    logger.debug(row[category])
    shap_explanation(model, row, category)


def shap_explanation(model: TransformerModel, data: pd.Series, label) -> None:
    masker = shap.maskers.Text(model.tokenizer)
    explainer = shap.Explainer(model.shap_predict, masker,
                               output_names=model.label_encoder.classes_.tolist())

    text = str(data["text"])
    logger.info(f"Explaining the model for : {text}")
    shap_values = explainer([text])

    with open(f"shap_values_{label}_{config['augmenter_name']}.pkl", "wb") as f:
        pickle.dump(shap_values, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="Type the path and name to the config file",
                        default='config.yaml')
    parser.add_argument('--category', type=str,
                        choices=['hazard-category', 'product-category', 'hazard', 'product'],
                        help="Type the category you want to check the missclassifications",
                        default='hazard-category')
    parser.add_argument('--dataname', type=str,
                        help="Type the name of the dataset to predict and make shap",
                        default='incidents_test_cleaned.csv')
    parser.add_argument("--index", type=int,
                        help="Type the index of the row to make the shap explanation",
                        default=25)
    args = parser.parse_args()
    config = tools.load_params(args.config)

    filepath = config["dataprocesseddirectory"] + args.dataname
    test_df = dataio.load_df(filepath)
    predict(config, args.category, test_df, args.index)