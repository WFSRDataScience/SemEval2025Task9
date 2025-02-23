import pandas as pd
import argparse

import src.common.tools as tools
import src.data.dataio as dataio
from src.evaluate.results import Results
from src import logging

logger = logging.getLogger(__name__)
pd.options.display.max_colwidth = 300

def get_results(results_path: str) -> Results:
    res = tools.pickle_load(results_path)
    results = Results(res.y_true, res.y_pred, res.classes)
    results.set_metrics()
    results.print_metrics()
    return results

def predict(config: dict, category: str, test_df: pd.DataFrame) -> None:
    resultspath_baseline = f'{config["resultsrawpath"]}{config["modelname"]}-{config["field"]}/baseline/results_raw_{category}.pkl'
    resultspath_con = f'{config["resultsrawpath"]}{config["modelname"]}-{config["field"]}/contextual_words/results_raw_{category}.pkl'

    results_baseline = get_results(resultspath_baseline)
    results_con = get_results(resultspath_con)

    rows_correct_in_baseline_not_cw = [i for i, (pred_base, pred_con, true) in enumerate(zip(results_baseline.y_pred,
                                                                                    results_con.y_pred,
                                                                                    results_baseline.y_true))
                                                                            if pred_base == true and pred_con != true]
    assert len(results_baseline.y_true) == len(results_con.y_true)

    logger.info("Indices of rows that are correct in baseline but misclassified in contextual words:")
    logger.info(rows_correct_in_baseline_not_cw)
    logger.info(test_df.iloc[rows_correct_in_baseline_not_cw])
    for ind in rows_correct_in_baseline_not_cw:
        logger.info(f"Index: {ind} | Correct label: {results_baseline.y_true[ind]} | Baseline prediction: {results_baseline.y_pred[ind]} | Contextual words prediction: {results_con.y_pred[ind]}")

    rows_correct_in_cw_not_baseline = [i for i, (pred_base, pred_con, true) in enumerate(zip(results_baseline.y_pred,
                                                                                results_con.y_pred,
                                                                                results_baseline.y_true))
                                                                            if pred_base != true and pred_con == true]

    logger.info("Indices of rows that are correct in contextual words but misclassified in baseline:")
    logger.info(rows_correct_in_cw_not_baseline)
    logger.info(test_df.iloc[rows_correct_in_cw_not_baseline])
    for ind in rows_correct_in_cw_not_baseline:
        logger.info(f"Index: {ind} | Correct label: {results_baseline.y_true[ind]} | Baseline prediction: {results_baseline.y_pred[ind]} | Contextual words prediction: {results_con.y_pred[ind]}")

if __name__ == "__main__":
    # load the data and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="Type the path and name to the config file",
                        default='config.yaml')
    parser.add_argument('--category', type=str,
                        choices=['hazard-category', 'product-category', 'hazard', 'product'],
                        help="Type the category you want to check the missclassifications",
                        default='hazard-category')
    args = parser.parse_args()
    config = tools.load_params(args.config)

    filepath = config["dataprocesseddirectory"] + "incidents_test_cleaned.csv"
    test_df = dataio.load_df(filepath)
    predict(config, args.category, test_df)