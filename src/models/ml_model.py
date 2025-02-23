from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

from src.common import tools
from src.models import Model
from src import logging
logger = logging.getLogger(__name__)

class MLModel(Model):
    NAMES = ['RF', 'SVM', 'NB', 'LR', 'KNN', 'DT']

    MODEL_PARAMS = {
        'RF': ['n_estimators', 'max_depth'],
        'SVM': ['C', 'max_iter'],
        'NB': ['alpha'],
        'LR': ['C', 'max_iter'],
        'KNN': ['n_neighbors', 'weights'],
        'DT': ['max_depth',]
    }

    def __init__(self, model_name:str, config: dict):
        # check if model_name is in NAMES
        if model_name not in MLModel.NAMES:
            raise ValueError(f"Model name {model_name} not supported")
        super().__init__(model_name, config)
        self.pipeline = None
        self.tf_idf_params = None
        self.model_params = None
        self.field = None
        self.category = None
        self.pipeline = None

    def set_params(self, params: dict, field:str, category:str) -> None:
        self.field = field
        self.category = category
        if 'max_features' not in params:
            params['max_features'] = None

        tokenizer = None
        stop_words = None
        if 'tokenizer' in params and params['tokenizer'] == 'spacy':
            tokenizer = tools.spacy_tokenizer
            stop_words = list(en_stop)

        self.tf_idf_params = {'strip_accents':'unicode',
                             'analyzer': params['analyzer'],
                             'max_features': params['max_features'],
                             'tokenizer': tokenizer,
                             'ngram_range': eval(params['ngram_range']),
                             'min_df': params['min_df'],
                             'max_df':params['max_df'],
                             'stop_words': stop_words}

        # Filter model-specific parameters
        valid_params = self.MODEL_PARAMS[self.model_name]
        self.model_params = {k: v for k, v in params.items() if k in valid_params}

    def get_model(self) -> BaseEstimator:
        if self.model_name == 'RF':
            return RandomForestClassifier(class_weight='balanced', random_state=self.config["seed"],
                                          n_jobs=-1, **self.model_params)
        elif self.model_name == 'SVM':
            return LinearSVC(class_weight='balanced', **self.model_params)
        elif self.model_name == 'NB':
            return MultinomialNB(**self.model_params)
        elif self.model_name == 'LR':
            return LogisticRegression(class_weight='balanced', n_jobs=-1, **self.model_params)
        elif self.model_name == 'KNN':
            return KNeighborsClassifier(n_jobs=-1, **self.model_params)
        elif self.model_name == 'DT':
            return DecisionTreeClassifier(class_weight='balanced',random_state=self.config["seed"], **self.model_params)

    def initialize_model(self, labels)-> None:
        tf_idf_vectorizer = TfidfVectorizer(**self.tf_idf_params)
        model = self.get_model()

        self.pipeline = Pipeline([
            ('tfidf', tf_idf_vectorizer),
            ('model', model)
        ])

    def initialize_model_predict(self):
        self.load()

    def train_test(self, train_df, augmented_df=None, valid_df=None) -> tuple:

        if augmented_df is not None:
            train_df = pd.concat([train_df, augmented_df], ignore_index=True)
            logger.info(f"Training on augmented data with shape {train_df.shape}")
        else:
            logger.info(f"Training on data with shape {train_df.shape}")

        X_train = train_df[self.config['field']]
        y_train = train_df[self.category].values
        if valid_df is not None:
            X_test = valid_df[self.config['field']]
            y_test = valid_df[self.category].values

        self.pipeline.fit(X_train, y_train)
        if valid_df is not None:
            y_pred = self.pipeline.predict(X_test)

            f1_macro = f1_score(y_test, y_pred, average='macro')
            acc = accuracy_score(y_test, y_pred)
            logger.info(f"F1 macro: {f1_macro} Accuracy: {acc}")
            return f1_macro, acc
        else:
            logger.info(f"Training on train and valid data completed")
            return None, None

    def predict_proba(self, test_df: pd.DataFrame) -> list:
        X = test_df[self.config['field']]
        prediction = self.pipeline.predict_proba(X)
        classes = self.pipeline.classes_
        return [prediction, classes]

    def predict(self, test_df: pd.DataFrame) -> list:
        X = test_df[self.config['field']]
        prediction = self.pipeline.predict(X)
        classes = self.pipeline.classes_
        return [prediction, classes]

    def save(self) -> None:
        path = f"{self.config['modelpath']}{self.model_name}-{self.field}/{self.config['augmenter_name']}/{self.category}/model.pkl"
        logger.info(f"Saving model to {path}")
        tools.pickle_dump(path, self.pipeline)

    def load(self) -> None:
        path = f"{self.config['modelpath']}{self.model_name}-{self.field}/{self.config['augmenter_name']}/{self.category}/model.pkl"
        logger.info(f"Loading model from {path}")
        self.pipeline = tools.pickle_load(path)
