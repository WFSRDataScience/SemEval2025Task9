import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pycm import ConfusionMatrix

import src.common.tools as tools
from src import logging
from src.evaluate.results import Results

logger = logging.getLogger(__name__)

def confusion(y_true: list, y_pred: list, labels: list) -> tuple:
    # Create a confusion matrix from the results
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax, normalize='true')

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    return fig, cm


def visualize_results(results: Results, outfolder: str, category: str) -> None:
    # Create and save all visualizations
    fig, cm = confusion(results.y_true, results.y_pred, results.classes)
    tools.check_and_create_path(outfolder)
    cm.save_html(figuredirectory + f"confusion_{category}")


if __name__ == "__main__":
    # Load the results
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="Type the path and name to the config file",
                        default='config.yaml')
    args = parser.parse_args()
    config = tools.load_params(args.config)

    categories = config["categories"]
    for category in categories:
        resultspath = f"{config['resultsevaluatedpath']}{config['modelname']}-{config['field']}/{config['augmenter_name']}/results_evaluated_{category}.pkl"
        results = tools.pickle_load(resultspath)

        # Produce and save the figure
        figuredirectory = config["figuredirectory"] + config["modelname"] + "/"
        logger.info(f'Saving visualization results for:{category}')
        visualize_results(results, figuredirectory, category)
