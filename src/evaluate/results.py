from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from src import logging
logger = logging.getLogger(__name__)

class Results:
    def __init__(self, y_true: list, y_pred: list, classes: list) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes
        self.metrics = {}

    def set_metrics(self) -> None:
        self.metrics['confusion_matrix'] = confusion_matrix(self.y_true, self.y_pred)
        self.metrics["accuracy"] = accuracy_score(self.y_true, self.y_pred)
        self.metrics["macro_f1_score"] = f1_score(self.y_true, self.y_pred, average="macro")
        self.metrics["micro_f1_score"] = f1_score(self.y_true, self.y_pred, average="micro")


    def print_metrics(self) -> None:
        for key in self.metrics:
            logger.info(f"{key} =\n {self.metrics[key]}")
