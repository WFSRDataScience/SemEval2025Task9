import os
import argparse
import pandas as pd

from src.common import tools
from src.data import dataio
from src import logging
from src.models import get_model_name

logger = logging.getLogger(__name__)

def make_submission(config: dict, test_df: pd.DataFrame) -> None:
    # get the example submission
    submission = dataio.load_df('data/submission.csv')

    submission = submission.iloc[0:0]

    categories = config["categories"]
    for category in categories:
        modelname = config['modelname']
        logger.info(f"Predicting model {modelname} for {category} for {len(test_df)} samples")

        model = get_model_name(modelname, config)

        model.set_params(config, config["field"], category)

        model.initialize_model_predict()
        [y_hat, classes] = model.predict(test_df)

        submission[category] = y_hat

    # Save the submissions
    csvname = 'submission.csv'
    # join the path and the filename
    csv_path = os.path.join(config["submissionpath"], config["modelname"] + "-" + config["field"], csvname)
    dataio.save_df(submission, csv_path, index=True)

    # Zip the submission
    zip_path = 'submission.zip'
    tools.zipdir(submission, csvname, zip_path)

if __name__ == "__main__":
    # load the data and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="Type the path and name to the config file",
                        default='config.yaml')
    parser.add_argument('--dataset', type=str,
                        help="Type the name of the dataset to predict and make submission for",
                        default='incidents_test_cleaned.csv')

    args = parser.parse_args()
    config = tools.load_params(args.config)
    dataset = args.dataset

    # Load the data to make predictions on
    filepath = config["dataprocesseddirectory"] + dataset
    test_df = dataio.load_df(filepath)
    make_submission(config, test_df)