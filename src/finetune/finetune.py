import argparse
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from src import logging, exception_handler
from src.common import tools
from src.data import dataio
from src.models import TransformerModel, get_model_name, MLModel

logger = logging.getLogger(__name__)

def get_trial_params(trial: optuna.Trial, model_name: str) -> dict:
    params = {}
    if model_name == 'RF':
        params.update({
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300]),
            'max_depth': trial.suggest_categorical('max_depth', [100, 1000, 5000]),
            'max_features': trial.suggest_categorical('max_features', [1000, 5000, 10000, 50000]),
        })
    elif model_name == 'SVM' or model_name == 'LR':
        params.update({
            'max_features' : trial.suggest_categorical('max_features', [1000, 5000, 10000, 50000]),
            'C': trial.suggest_categorical('C', [0.1, 1, 5, 10]),
            'max_iter': trial.suggest_categorical('max_iter', [100, 1000, 5000]),
        })
    elif model_name == 'NB':
        params.update({
            'alpha': trial.suggest_categorical('alpha', [0.01, 0.1, 1, 5]),
        })
    elif model_name == 'KNN':
        params.update({
            "n_neighbors" : trial.suggest_categorical('n_neighbors', [3, 5, 7, 9, 11]),
            "weights" : trial.suggest_categorical('weights', ['uniform', 'distance']),
        })
    elif model_name == 'DT':
        params.update({
            'max_depth': trial.suggest_categorical('max_depth', [100, 200, 300]),
            'max_features': trial.suggest_categorical('max_features', [1000, 5000, 10000, 50000]),
        })
    elif model_name in TransformerModel.NAMES:
        params.update({
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', ['linear', 'cosine', 'cosine_with_restarts']),
            'epochs': trial.suggest_categorical('epochs', [3,5,10]),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        })
    else:
        raise ValueError(f"Model {model_name} not supported")

    if model_name not in TransformerModel.NAMES:
        # Define the hyperparameters for TF-IDF vectorizer
        analyzer = trial.suggest_categorical('analyzer', ['word', 'char'])
        tokenizer_option = trial.suggest_categorical('tokenizer', ['none', 'spacy']) if analyzer == 'word' else 'none'
        ngram_range = trial.suggest_categorical('ngram_range', ["(1, 1)", "(1, 2)", "(1, 3)", "(1, 4)",
                                                                 "(1, 5)", "(2, 3)", "(2, 4)", "(2, 5)","(3, 5)"])
        params.update({
            'ngram_range': ngram_range,
            'min_df': trial.suggest_categorical('min_df', [1, 2, 5]),
            'max_df': trial.suggest_categorical('max_df', [0.1, 0.3, 0.5]),
            'analyzer': analyzer,
            'tokenizer': tokenizer_option,
        })

    return params

def objective(trial: optuna.Trial, config: dict, train_df: pd.DataFrame, valid_df: pd.DataFrame,
              augmented_df: pd.DataFrame, category: str, field: str, model_name: str) -> float:

    model = get_model_name(model_name, config)

    # Define the hyperparameter space
    params = get_trial_params(trial, model_name)
    logger.info(f"Trial params: {params}")
    model.set_params(params, field, category)
    labels = pd.concat([train_df, valid_df], ignore_index=True)[category]
    model.initialize_model(labels)
    try:
        f1_macro_trial, _ = model.train_test(train_df=train_df, valid_df=valid_df, augmented_df=augmented_df)
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise optuna.TrialPruned()

    return f1_macro_trial

@exception_handler
def finetune_with_optuna(config: dict, category: str, field:str,
                         model_name:str, augmenter_name:str, n_trials: int) -> float:

    # Load data
    train_df = dataio.load_df(f"{config['dataprocesseddirectory']}incidents_train_cleaned.csv")
    valid_df = dataio.load_df(f"{config['dataprocesseddirectory']}incidents_valid_cleaned.csv")

    augmented_df = None
    if augmenter_name != 'baseline':
        augmented_path = f"{config['dataaugmenteddirectory']}incidents_train_{category}_{augmenter_name}_augmented.csv"
        augmented_df = dataio.load_df(augmented_path)

    study = optuna.create_study(direction='maximize',sampler=TPESampler())

    study.optimize(
        lambda trial: objective(trial, config, train_df, valid_df, augmented_df, category, field, model_name),
        n_trials=n_trials
    )

    # Save the best params to a file
    save_best_params(study, model_name, augmenter_name, category, field)
    return study.best_trial.value

def save_best_params(study, model_name: str, augmenter_name: str, label: str, field:str) -> None:
    filename = f"params/{model_name}/{augmenter_name}/{label}-{field}-best_params.txt"
    tools.check_and_create_path(filename)
    logger.info(f"Best params for {label} - {field} saved to file: {filename} "
                f"with best f1 macro of {study.best_trial.value}")

    with open(filename, "w") as f:
        f.write(str(study.best_trial.params))

if __name__ == '__main__':
    # load the data and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str, choices=['title', 'text'],
                        help="Type the type of field you want to use from the dataset to finetune. Default is 'title'",
                        default='title')
    parser.add_argument('--category', type=str,
                        choices=['product-category', 'hazard-category', 'product', 'hazard'],
                        help="Type the category you want to finetune the model for. Default is 'hazard-category'",
                        default='hazard-category')
    parser.add_argument('--model_name', type=str,
                        choices = TransformerModel.NAMES + MLModel.NAMES,
                        help="Type the name of the model to use. Default is 'RF'",
                        default='RF')
    parser.add_argument('--augmenter_name', type=str,
                        choices = ['baseline', 'contextual_words', 'synonym_words', 'random_words'],
                        help="Type the name of the augmenter to use. Default is 'baseline'",
                        default='baseline')
    parser.add_argument('--n_trials', type=int,
                        help="Type the number of trials to run. Default is 10",
                        default=10)
    parser.add_argument('--config', type=str,
                        help="Type the path and name to the config file",
                        default='config.yaml')
    args = parser.parse_args()

    config = tools.load_params(args.config)

    finetune_with_optuna(config, args.category, args.field, args.model_name, args.augmenter_name, args.n_trials)