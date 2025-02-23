import argparse

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.common import tools
from src.evaluate.results import Results
from src import logging, exception_handler
logger = logging.getLogger(__name__)

def plot_cm(ax, y_true: list, y_pred: list, classes: list, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='g', cmap=sns.cubehelix_palette(as_cmap=True), square=True,
                ax=ax, cbar=False, xticklabels=classes, yticklabels=classes, annot_kws={"size": 16})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)


def map_to_classes(y_true: list, y_pred: list, minority_classes: list) -> tuple[list, list]:
    mapped_y_true = ['Minority' if label in minority_classes else 'Majority' for label in y_true]
    mapped_y_pred = ['Minority' if label in minority_classes else 'Majority' for label in y_pred]
    return mapped_y_true, mapped_y_pred

@exception_handler
def plot_confusion_matrix(config: dict) -> None:
    class_names = ['Minority', 'Majority']
    categories_list = config["categories"]

    results = {}
    for category in categories_list:
        resultspath_baseline = f"{config['resultsrawpath']}bert-text/baseline/results_raw_{category}.pkl"
        resultspath_augmented = f"{config['resultsrawpath']}/bert-text/contextual_words/results_raw_{category}.pkl"
        results_baseline = tools.pickle_load(resultspath_baseline)
        results_augmented = tools.pickle_load(resultspath_augmented)

        r_b = Results(results_baseline.y_true, results_baseline.y_pred, results_baseline.classes)
        r_a = Results(results_augmented.y_true, results_augmented.y_pred, results_augmented.classes)

        r_b.set_metrics()
        r_a.set_metrics()

        r_b.print_metrics()
        r_a.print_metrics()

        #get the minority classes by loading the augmented data and get the classes
        df_augment = pd.read_csv(f'{config["dataaugmenteddirectory"]}incidents_train_{category}_contextual_words_augmented.csv')
        minority_classes = df_augment.groupby(category).count().index.tolist()

        # map the minority and the majority classes
        y_true_mapped, y_pred_baseline_mapped = map_to_classes(r_b.y_true, r_b.y_pred,
                                                               minority_classes)
        _, y_pred_augmented_mapped = map_to_classes(r_b.y_true, r_a.y_pred, minority_classes)

        results[category] = {
            'y_true': y_true_mapped,
            'y_pred_baseline': y_pred_baseline_mapped,
            'y_pred_augmented': y_pred_augmented_mapped
        }

    n_rows = len(categories_list)
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.1, 8))

    for i, cat in enumerate(categories_list):
        plot_cm(axes[i, 0],
                results[cat]['y_true'],
                results[cat]['y_pred_baseline'],
                classes=class_names,
                title=f"{cat}" + "- $BERT_{base}$")

        plot_cm(axes[i, 1],
                results[cat]['y_true'],
                results[cat]['y_pred_augmented'],
                classes=class_names,
                title=f"{cat}" + "- $BERT_{CW}$")
    plt.tight_layout()
    # plt.show()
    plt.savefig('confusion_matrix_all.png')


if __name__ == '__main__':
    # load the data and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="Type the path and name to the config file",
                        default='config.yaml')
    args = parser.parse_args()
    config = tools.load_params(args.config)
    plot_confusion_matrix(config)

