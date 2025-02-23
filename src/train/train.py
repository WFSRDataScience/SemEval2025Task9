import pandas as pd
import argparse

from src import logging
import src.common.tools as tools
import src.data.dataio as dataio
from src import exception_handler
from src.models import get_model_name
logger = logging.getLogger(__name__)

@exception_handler
def train(config: dict, category: str)-> None:
    train_df = dataio.load_df(config["dataprocesseddirectory"] + "incidents_train_cleaned.csv", index=False)
    valid_df = dataio.load_df(config["dataprocesseddirectory"] + "incidents_valid_cleaned.csv")
    augmented_df = None

    augmenter_name = config["augmenter_name"]
    if augmenter_name != "baseline":
        logger.info(f"Loading augmented data for {augmenter_name}")
        augmented_df = dataio.load_df(
            config["dataaugmenteddirectory"] + f"incidents_train_{category}_{augmenter_name}_augmented.csv",
            index=False)

    modelname = config['modelname']
    model = get_model_name(modelname, config)
    labels = pd.concat([train_df, valid_df], ignore_index=True)[category] # 10 labels in product are missing in the train_df
    if config["use_best_params"]:
        best_params = dataio.read_best_params(f'params/{modelname}/{config["augmenter_name"]}/{category}-{config["field"]}-best_params.txt')
        model.set_params(best_params, config["field"], category)
    else:
        model.set_params(config, config["field"], category)
    model.initialize_model(labels)

    if not config["train_on_full_data"]:
        logger.info("Training on train and test on valid data")
        model.train_test(train_df=train_df, valid_df=valid_df, augmented_df=augmented_df)
    else:
        logger.info("Training on train and valid data")
        merged_df = pd.concat([train_df, valid_df], ignore_index=True)
        model.train_test(train_df=merged_df, augmented_df=augmented_df)

    model.save()

if __name__ == "__main__":
    # load the data and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="Type the path and name to the config file",
                        default='config.yaml')
    args = parser.parse_args()
    config = tools.load_params(args.config)

    if config["category"] == "all":
        categories = config["categories"]
        for cat in categories:
            train(config, cat)
    else:
        train(config, config["category"])
