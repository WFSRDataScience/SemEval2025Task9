import os
import yaml
import pickle
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from transformers import AutoModelForSequenceClassification, AutoTokenizer

nlp = English()
stop_words = list(en_stop)

def spacy_tokenizer(text):
    return [token.text for token in nlp(text)]

def load_params(config):
    with open(config) as p:
        config = yaml.safe_load(p)
    return config

def save_model(model, path):
    check_and_create_path(path)
    model.save_pretrained(path)

def save_tokenizer(tokenizer, path):
    check_and_create_path(path)
    tokenizer.save_pretrained(path)

def pickle_dump(path, variable):
    # Serialize data from memory to file
    check_and_create_path(path)
    with open(path, 'wb') as handle:
        pickle.dump(variable, handle)

def pickle_load(path):
    # Read and load serialized data from file
    with open(path, 'rb') as handle:
        loaded = pickle.load(handle)
    return loaded

def check_and_create_path(path):
    # remove the filename from the path
    path = os.path.dirname(path)

    if not check_path(path):
        create_path(path)

def check_path(path):
    # Check if the path exists
    return os.path.exists(path)

def create_path(path):
    # Create a new path
    os.makedirs(path)
    return path

def zipdir(df, path, zip_path):
    compression_opts = dict(method='zip',
                            archive_name=path)
    df.to_csv(zip_path, compression=compression_opts)

def load_model(path):
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model

def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    return tokenizer
