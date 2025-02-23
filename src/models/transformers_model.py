import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
import os

from src.common import tools
from src.data import dataio
from src.models import Model
from src import logging
logger = logging.getLogger(__name__)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

class TransformerModel(Model):
    NAMES = ['bert', 'roberta', 'distilbert', 'modernbert']

    def __init__(self, model_name: str, config: dict):
        if model_name not in TransformerModel.NAMES:
            raise ValueError(f"Model name {model_name} not supported")
        super().__init__(model_name, config)
        self.pretrained_model = self.get_pretrained_model()
        self.device = config["device"]

        self.label_encoder = None
        self.model = None
        self.tokenizer = None

    def set_params(self, params: dict, field: str, category: str):
        self.field = field
        self.category = category
        self.config.update(params)

    def get_pretrained_model(self) -> str:
        if self.model_name == 'bert':
            return 'bert-base-uncased'
        elif self.model_name == 'roberta':
            return 'roberta-base'
        elif self.model_name == 'distilbert':
            return 'distilbert-base-uncased'
        elif self.model_name == 'modernbert':
            return 'answerdotai/ModernBERT-base'
        else:
            raise ValueError(f"Model name {self.model_name} not supported")

    def initialize_model(self, labels) -> None:
        self.build_tokenizer()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit_transform(labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model,
                                                                        num_labels=len(labels.unique()))

    def initialize_model_predict(self):
        self.build_tokenizer()
        self.load()

    def build_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

    def tokenize_function(self, samples) -> AutoTokenizer:
        return self.tokenizer(samples[self.config["field"]], padding=True,
                              truncation=True, max_length=self.config["max_length"])

    def tokenize_function_shap(self, samples) -> AutoTokenizer:
        return self.tokenizer(samples[self.config["field"]].tolist(), padding=True,
                              truncation=True, max_length=self.config["max_length"])

    def build_dataloader(self, df, columns, shuffle=True) -> DataLoader:
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self.tokenize_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)

        dataset.set_format(type='torch', columns=columns)
        dataloader = DataLoader(dataset,
                                shuffle=shuffle,
                                batch_size=self.config["batch_size"],
                                collate_fn=data_collator)
        return dataloader

    def train_test(self, train_df, valid_df=None, augmented_df=None):

        if augmented_df is not None:
            train_df = pd.concat([train_df, augmented_df], ignore_index=True)
            logging.info(f"Training on augmented data with shape {train_df.shape}")
        else:
            logging.info(f"Training on data with shape {train_df.shape}")

        # Encode the labels
        train_df['label'] = self.label_encoder.transform(train_df[self.category])

        # Build the dataloaders
        columns = ['input_ids', 'attention_mask', 'label']
        train_dataloader = self.build_dataloader(train_df, columns)

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.config["lr"])
        num_epochs = self.config["epochs"]

        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name=self.config["lr_scheduler"],
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        # Training loop
        self.model.to(self.device)
        self.model.train()

        progress_bar = tqdm(range(num_training_steps))
        train_last_loss = 0
        valid_f1, valid_accuracy = 0, 0
        for epoch in range(num_epochs):
            for i,batch in enumerate(train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # pass the batch through the model
                outputs = self.model(**batch)
                loss = outputs.loss

                # calculating the gradients for the loss function
                loss.backward()

                # optimizing the parameters of the model
                optimizer.step()

                # updating the learning rate
                lr_scheduler.step()

                # setting the gradients to zero
                optimizer.zero_grad()

                # Calculating the running loss for logging purposes
                train_batch_loss = loss.item()
                train_last_loss = train_batch_loss / self.config["batch_size"]

                progress_bar.update(1)

            logging.info(f"\nTraining epoch {epoch + 1} loss: {train_last_loss}")
            # Training loop ends

            # Validation starts
            test_last_loss = 0
            valid_f1, valid_accuracy = 0, 0
            if valid_df is not None:
                # Encode labels
                valid_df['label'] = self.label_encoder.transform(valid_df[self.category])

                # Build the dataloader
                valid_dataloader = self.build_dataloader(valid_df, columns, shuffle=False)

                # Set model to evaluation mode
                self.model.eval()
                acc, f1 = [] ,[]
                val_preds, val_true = [], []
                for i, batch in enumerate(valid_dataloader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    with torch.no_grad():
                        outputs = self.model(**batch)

                    loss = outputs.loss
                    if loss is not None:
                        test_batch_loss = loss.item()

                    # Calculating the mean batch loss
                    test_last_loss = test_batch_loss / self.config["batch_size"]

                    logits = outputs.logits
                    val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                    val_true.extend(batch['labels'].cpu().numpy())
                valid_accuracy = accuracy_score(val_true, val_preds)
                valid_f1 = f1_score(val_true, val_preds, average='macro')
                logging.info(f"Validation Accuracy: {valid_accuracy}, Validation F1 Macro: {valid_f1}")
                acc.append(valid_accuracy)
                f1.append(valid_f1)

                logging.info(f"\nTesting epoch {epoch + 1} last loss: {test_last_loss} ")

        return valid_f1, valid_accuracy

    def predict(self, test_df: pd.DataFrame) -> list:
        columns = ['input_ids', 'attention_mask']
        # Build the dataloader
        test_dataloader = self.build_dataloader(test_df, columns, shuffle=False)

        progress_bar = tqdm(test_dataloader)

        self.model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions_ = torch.argmax(outputs.logits, dim=-1)
                predictions.extend([p.item() for p in predictions_])
                progress_bar.update(1)

        # Decode the labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        # gold_labels = self.label_encoder.inverse_transform(test_df.label.values)
        classes = self.label_encoder.classes_
        return [predicted_labels, classes]

    def shap_predict(self, texts: list) -> list:

        df = pd.DataFrame({self.config["field"]: texts})
        tokenized_inputs = self.tokenize_function_shap(df)
        tokenized_inputs = {key: torch.tensor(val).to(self.device)
                            for key, val in tokenized_inputs.items()}
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs).logits
            predictions = torch.argmax(outputs, dim=-1)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        logger.debug(f"Output probabilities: {outputs}")
        logger.debug(f"Predicted labels: {predicted_labels}")
        return outputs.detach().cpu().numpy()

    def save(self):
        path = f'{self.config["modelpath"]}{self.config["modelname"]}-{self.config["field"]}/{self.config["augmenter_name"]}'
        logging.info(f"Saving model, tokenizer and label encoder to {path}")

        tools.save_model(self.model, path + f'/model_{self.category}')
        tools.save_tokenizer(self.tokenizer, path + f'/tokenizer_{self.category}')
        tools.pickle_dump(path + f'/labelencoder_{self.category}.pkl', self.label_encoder)

    def load(self):
        path = f'{self.config["modelpath"]}{self.config["modelname"]}-{self.config["field"]}/{self.config["augmenter_name"]}'
        logging.info(f"Loading model, tokenizer and label encoder from {path}")

        self.tokenizer = tools.load_tokenizer(path + f'/tokenizer_{self.category}')
        self.model = tools.load_model(path + f'/model_{self.category}')
        self.label_encoder = tools.pickle_load(path + f'/labelencoder_{self.category}.pkl')

