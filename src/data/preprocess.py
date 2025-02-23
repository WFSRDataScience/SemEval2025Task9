from src import logging
import re
import argparse
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.common import tools
from src.data import dataio
logger = logging.getLogger(__name__)

tqdm.pandas()
pd.options.display.max_colwidth = 390 # Increase column width to display more text

# Regular expression to remove special characters
REG_EXP = r'[\t\n\r\u200b]|//|&nbsp'
SPECIAL_CHARS_PATTERN = re.compile(REG_EXP)

def remove_html_tags(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text()

def has_html_tags(text: str) -> bool:
    if pd.isnull(text):
        return False
    return BeautifulSoup(text, "html.parser").text != text

def remove_special_chars(text: str) -> str:
    text = re.sub(REG_EXP, ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def has_special_chars(text: str) -> bool:
    if pd.isnull(text):
        return False
    return bool(SPECIAL_CHARS_PATTERN.search(str(text)))

def clean_data(config: dict, dataname: str) -> None:
    logger.info(f"Cleaning csv dataset for: {config['datarawdirectory']}{dataname}")
    base_name = dataname.split('.')[0]

    try:
        df_raw = dataio.load_df(f"{config['datarawdirectory']}{dataname}")

        # Check for HTML tags
        html_counts_title = df_raw['title'].apply(has_html_tags).sum()
        html_counts_text = df_raw['text'].apply(has_html_tags).sum()
        logger.info(f"Number of rows with HTML tags in 'title': {html_counts_title} and 'text' :{html_counts_text}")

        # Remove HTML tags
        df_raw['title'] = df_raw['title'].progress_apply(lambda x: remove_html_tags(x))
        df_raw['text'] = df_raw['text'].progress_apply(lambda x: remove_html_tags(x))

        # Check for special characters
        special_chars_counts_title = df_raw['title'].apply(has_special_chars).sum()
        special_chars_counts_text = df_raw['text'].apply(has_special_chars).sum()
        logger.info(f"Number of rows with special char in 'title': {special_chars_counts_title} "
              f"and 'text' :{special_chars_counts_text}")

        # Remove special characters
        df_raw['title'] = df_raw['title'].progress_apply(lambda x: remove_special_chars(x))
        df_raw['text'] = df_raw['text'].progress_apply(lambda x: remove_special_chars(x))

        dataio.save_df(df_raw, f"{config['dataprocesseddirectory']}{base_name}_cleaned.csv", index=True)

    except FileNotFoundError as e:
        logger.error(f"File opening failed: {str(e)}")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")

if __name__ == '__main__':

    # load the data and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str,
                        help="Type the csv dataset name to clean. Make sure it is on the "
                             "datarawdirectory configured in the config.yaml",
                        default='incidents_test.csv')
    parser.add_argument('--config', type=str,
                        help="Type the path and name to the config file",
                        default='config.yaml')
    args = parser.parse_args()

    dataname = args.dataname
    config = tools.load_params(args.config)

    # Clean the data
    clean_data(config, dataname)




