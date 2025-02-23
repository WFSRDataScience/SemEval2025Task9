import argparse

from src.common import tools
from src.common.metrics import compute_score
from src.evaluate.results import Results
from src import logging, exception_handler

logger = logging.getLogger(__name__)

@exception_handler
def store_evaluated_results(config: dict, path: str) -> None:
    categories = config["categories"]

    for category in categories:
        # Load results
        resultspath = path + f"/results_raw_{category}.pkl"
        results = tools.pickle_load(resultspath)
        res = Results(results.y_true, results.y_pred, results.classes)

        # Calculate metrics
        res.set_metrics()
        logger.info(f"Metrics for {category}")
        res.print_metrics()

        # Save metrics
        validationpath = f'{config["resultsrawpath"]}{config["modelname"]}-{config["field"]}/{config["augmenter_name"]}/results_evaluated_{category}.pkl'
        tools.pickle_dump(validationpath, res)

@exception_handler
def print_scores(resultspath: str, hazard_label: str, product_label: str, task_label: int) -> float:
    resultspath_hazard = resultspath + f"/results_raw_{hazard_label}.pkl"
    resultspath_product = resultspath + f"/results_raw_{product_label}.pkl"

    res_hazard = tools.pickle_load(resultspath_hazard)
    res_product = tools.pickle_load(resultspath_product)
    score = compute_score(res_hazard.y_true, res_product.y_true, res_hazard.y_pred, res_product.y_pred)
    logger.info(f'Score Sub-Task {task_label}: {score}')
    return score


if __name__ == "__main__":
    # load the data and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="Type the path and name to the config file",
                        default='config.yaml')
    args = parser.parse_args()
    config = tools.load_params(args.config)

    path = f'{config["resultsrawpath"]}{config["modelname"]}-{config["field"]}/{config["augmenter_name"]}'

    print_scores(path, 'hazard-category', 'product-category', 1)
    print_scores(path, 'hazard', 'product', 2)

    store_evaluated_results(config, path)
