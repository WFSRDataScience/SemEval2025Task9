import logging
import argparse
import random
import numpy as np
import torch

from src.common import tools

# Configure logging level
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    help="Type the path and name to the config file",
                    default='config.yaml')
args, unknown = parser.parse_known_args()
config = args.config
config = tools.load_params(config)

logging.info("Seeding #####################")
torch.cuda.manual_seed(config['seed'])
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])

def exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception: {str(e)}")
    return inner_function