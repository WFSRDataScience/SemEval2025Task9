import pandas as pd
import argparse

from src.evaluate import evaluate
import src.common.tools as tools
import src.data.dataio as dataio
from src.models import get_model_name
from src import logging
logger = logging.getLogger(__name__)

def predict(config: dict, category: str, test_df: pd.DataFrame) -> None:

    modelname = config['modelname']
    logger.info(f"Predicting for model {modelname} and category: {category}")

    model = get_model_name(modelname, config)

    model.set_params(config, config["field"], category)

    model.initialize_model_predict()
    [y_hat, classes] = model.predict(test_df)

    # Save the results
    results = evaluate.Results(test_df[category], y_hat, classes)

    resultspath = f'{config["resultsrawpath"]}{config["modelname"]}-{config["field"]}/{config["augmenter_name"]}/results_raw_{category}.pkl'
    tools.pickle_dump(resultspath, results)


if __name__ == "__main__":
    # load the data and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="Type the path and name to the config file",
                        default='config.yaml')
    parser.add_argument('--dataname', type=str,
                        help="Type the name of the data file to predict your model",
                        default='incidents_test_cleaned.csv')
    args = parser.parse_args()
    dataname = args.dataname
    config = tools.load_params(args.config)

    test_df = dataio.load_df(config["dataprocesseddirectory"] + dataname)
    if config["category"] == "all":
        categories = config["categories"]
        for cat in categories:
            predict(config, cat, test_df)
    else:
        predict(config, config["category"], test_df)
