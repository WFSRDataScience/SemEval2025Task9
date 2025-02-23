import argparse
import pandas as pd
from src.data import dataio
from sklearn.utils import shuffle
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import nltk
nltk.download('averaged_perceptron_tagger_eng')

from src import logging, exception_handler
import src.common.tools as tools
from src import config
logger = logging.getLogger(__name__)


aug_CW = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert", device=config['device'])
aug_RANDOM = naw.RandomWordAug(action="swap")
aug_SYN = naw.SynonymAug(aug_src='wordnet')

def augment_with_random_words(text: str, n: int) -> list[str]:
    augmented_text = aug_RANDOM.augment(text, n=n)
    return augmented_text

def augment_with_contextual_words(text: str, n: int) -> list[str]:
    augmented_text = aug_CW.augment(text, n=n)
    return augmented_text

def augment_with_synonym(text: str, n: int) -> list[str]:
    augmented_text = aug_SYN.augment(text, n)
    return augmented_text

def augment_row(row: pd.DataFrame , n: int, augmenter_name: str) -> tuple[list[str], list[str]]:
    logger.debug(f"Augmenting row with title: {row['title']} and text: {row['text']} for {n} times")
    logger.debug(f"Augmenting row: {row.index}  for {n} times")
    if augmenter_name == 'contextual_words':
        rows_title = augment_with_contextual_words(row['title'], n)
        rows_text = augment_with_contextual_words(row['text'], n)
    elif augmenter_name == 'synonym':
        rows_title = augment_with_synonym(row['title'], n)
        rows_text = augment_with_synonym(row['text'], n)
    elif augmenter_name == 'random_words':
        rows_title = augment_with_random_words(row['title'], n)
        rows_text = augment_with_random_words(row['text'], n)
    else:
        raise ValueError("No augmenter found")

    return rows_title, rows_text

def count_classes_size(df: pd.DataFrame, columns: list) -> pd.Series:
    class_counts = df.groupby(columns).size()
    return class_counts

@exception_handler
def augment_low_supported_classes(train_df: pd.DataFrame, category: str, augmenter_name: str,
                                  threshold: int, samples_to_add: int) -> pd.DataFrame:
    """
    Augment the classes that have less than a certain threshold
    based on the samples_to_add in the config file
    """
    # check for classes that they have values less than a certain threshold
    class_counts = count_classes_size(train_df, [category])
    logger.info(class_counts.describe())

    # find the classes that have less than the threshold
    low_supported_classes = class_counts[class_counts < threshold]
    low_supported_classes = low_supported_classes.reset_index(name='class_count')

    augmented_df = pd.DataFrame(columns=train_df.columns)

    # for each class, augment the rows
    for index, row in tqdm(low_supported_classes.iterrows(), total=low_supported_classes.shape[0]):

        low_supported_class = row[category]
        low_supported_rows = train_df[train_df.apply(lambda x: x[category] == low_supported_class, axis=1)].copy()
        low_supported_rows = shuffle(low_supported_rows) # just shuffle them to see the effect on different classes

        index_ = 0
        while True:

            row_ = low_supported_rows.iloc[index_]
            row_copy = row_.copy()

            numb_rows_to_augment = int(samples_to_add//low_supported_rows.shape[0])
            if index_ == low_supported_rows.shape[0]-1:
                numb_rows_to_augment = samples_to_add - (numb_rows_to_augment * (low_supported_rows.shape[0]-1))
            rows_title, rows_text = [], []
            if numb_rows_to_augment > 0 :
                rows_title, rows_text = augment_row(row_, numb_rows_to_augment, augmenter_name=augmenter_name)

            # add augmented rows to the augmented_df
            for i in range(len(rows_text)):
                row_copy['title'] = rows_title[i]
                row_copy['text'] = rows_text[i]
                augmented_df.reset_index(drop=True, inplace=True)
                augmented_df = pd.concat([augmented_df,pd.DataFrame.from_records([row_copy.to_dict().copy()])])
            logger.debug(f"Augmented_df shape: {augmented_df.shape}")
            if index_ >= low_supported_rows.shape[0]-1:
                break

            index_ += 1

    # merge train_df and augmented_df
    train_df_augmented = pd.concat([train_df,augmented_df], axis=0)
    logger.info(f"Train_df shape after augmentation: {train_df_augmented.shape}")

    class_counts = count_classes_size(train_df_augmented, [category])
    logger.info("New class counts with augmented data:")
    logger.info(class_counts.describe())

    # save the augmented_df
    augmented_df.to_csv(config["dataaugmenteddirectory"] + f"incidents_train_{category}_{augmenter_name}_augmented.csv", index=False)

    return augmented_df

if __name__ == '__main__':

    # load the data and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="Type the path and name to the config file",
                        default='config.yaml')
    parser.add_argument('--category', type=str,
                        choices=['hazard-category', 'product-category', 'hazard', 'product'],
                        help="Type the category to augment",
                        default='hazard-category')
    parser.add_argument('--augmenter_name', type=str,
                        choices=['contextual_words', 'synonym', 'random_words'],
                        help="Type the augmenter name",
                        default='contextual_words')
    parser.add_argument('--threshold', type=int,
                        help="Type the threshold to augment the classes",
                        default=200)
    parser.add_argument('--samples_to_add', type=int,
                        help="Type the number of samples to add for each class",
                        default=200)

    args = parser.parse_args()
    category = args.category
    augmenter_name = args.augmenter_name
    threshold = args.threshold
    samples_to_add = args.samples_to_add
    config = tools.load_params(args.config)

    train_df = dataio.load_df(config["dataprocesseddirectory"] + "incidents_train_cleaned.csv")

    augmented_trained_df = augment_low_supported_classes(train_df, category, augmenter_name, threshold, samples_to_add)
