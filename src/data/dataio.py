import joblib
import pandas as pd
import ast

from src.common import tools
from src import logging
logger = logging.getLogger(__name__)

def load_df(datapath: str, index=True):
    if index:
        dataset = pd.read_csv(datapath, index_col=0)
    else:
        dataset = pd.read_csv(datapath)
    return dataset

def save_df(dataframe, save_path, index=False):
    logger.info(f"Saving to {save_path}")
    tools.check_and_create_path(save_path)
    dataframe.to_csv(save_path, index=index)

def save_pipeline(pipeline, pipeline_path):
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Pipeline saved at {pipeline_path}")

def read_best_params(path):
    with open(path, 'r') as file:
        data = file.read()

    dictionary = ast.literal_eval(data)
    return dictionary
