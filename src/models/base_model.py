import pandas as pd


class Model:
    def __init__(self, model_name:str, config: dict):
        self.model_name = model_name
        self.config = config
        self.field = None
        self.category = None

    def set_params(self, params: dict, field:str, label:str):
        pass

    def initialize_model(self, labels):
        pass

    def train_test(self, train_df, augmented_df=None, valid_df=None):
        pass

    def train(self):
        pass

    def predict(self, test_df: pd.DataFrame) -> list:
        pass

    def save(self):
        pass

    def load(self):
        pass

    def initialize_model_predict(self):
        pass

