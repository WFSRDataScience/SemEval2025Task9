# BrightCookies at SemEval-2025 Task 9: Exploring Data Augmentation for Food Hazard Classification

This repository contains the code for the BrightCookies team 
submission to the SemEval-2025 Task 9: Food Hazard Detection Challenge.
In this work, we propose text augmentation techniques as a way to improve
poor performance in minority classes, and compare their effect for each 
category on various encoder-only and traditional machine learning algorithms.
## Installation

To run the code, you need to have installed `python>=3.10.16` and install the required packages in the `requirements.txt`.

You can create a new conda environment and install the dependencies with the following commands:
```bash
conda create -y -n semeval python=3.10.16
conda activate semeval
pip install -r requirements.txt 
```
## Project Structure
The project structure is as follows:


```
.
├── data │ 
         ├── augmented   # augmented datasets
         ├── processed   # cleaned datasets
         └── raw         # raw datasets
├── docker               # docker files to run the code
├── notebooks            # jupyter notebooks for analysis
├── params               # best hyperparameters found by Optuna for each model
├── results              # results of the models. Unzip the results.zip file to get some results if you don't want to train a model
├── src │ 
        ├── analysis     # scripts for error analysis
        ├── common  
        ├── data         # preprocessing and augmentation scripts
        ├── evaluate     # prediction and evaluation scripts
        ├── finetune     
        ├── models       # ml and transformer models 
        └── train 
├── config.yaml          # configuration file to use for training and evaluation
└── README.md
```

## Running the Code
To reproduce the results, you should clone this repo and  run the following steps for each category and model you want to augment and evaluate.

Unzip the `data/augmented/incidents_train_augmented.zip` file to get the augmented datasets in case you don't want to run the 
code to generate them. Similarly, you can unzip `results/results.zip` file to
get some results from models that predict the test sets.
### Preprocessing Pipeline

The `src/data/preprocess.py` script cleans raw text data by removing HTML tags and special characters (e.g., newlines, tabs, Unicode artifacts) 
using `BeautifulSoup` and regular expressions. 
It processes CSV datasets, specifically the 
`title` and `text` columns, and saves the cleaned data 
to a new CSV file.

#### Example Usage:
Run the script with the following command:
```bash
python -m src.data.preprocess --config config.yaml --dataname incidents_train.csv 
```

Clean the three sets (train, valid, test) that are in the `data/raw` directory.

### Data Augmentation Pipeline
The `src/data/augment.py` script creates the augmented data by 
applying text augmentation techniques to the cleaned data based on the `category` you want to apply it.
The available 
augmentation techniques are contextual words insertion using `bert-base-uncased` (`contextual_words`),
random word swapping (`random_words`), synonym substitution using wordNet (`synonym`).
The script has the following parameters:
* `config`: the configuration file that contains general parameters such as `device` to run the code.
* `category`: the category you want to augment (e.g. `hazard-category`)
* `augmenter_name`: the name of the augmenter you want to use (e.g. `synonym`)
* `threshold`: a threshold to apply for the classes that has under this number of samples 
* `samples_to_add`: the number of samples to generate for each class
 to determine the classes you want to augment (based on the number of samples in the class)



#### Example Usage:
Run the script with the following command:
```bash
python -m src.data.augment --config config.yaml --category hazard-category --augmenter_name synonym --threshold 200 --samples_to_add 200
```

Augment the train set for all the four categories and for all the three available augmenters.

### Training Pipeline

The `src/train/train.py` script trains the selected model based on the `config.yaml` file and 
the parameters you want to use. 

Check the `config.yaml` file to see all the available parameters you can use to train the model you choose based on the 
`modelname` in `config.yaml`.
#### Example Usage:
Run the script with the following command:
```bash
python -m src.train.train --config config.yaml
```
Run the train script for all the models, categories and augmenters possible combinations.
### Evaluation Pipeline
The `src/evaluate/predict.py` script predicts the model saving the predictions.
This script can get the following parameters:
* `config`: the configuration file that contains general parameters such as `device` to run the code.
* `dataname`: the name of the dataset you want to predict the model on and is on the `data/processed` folder.


The `src/evaluate/evaluate.py` script evaluates the predictions by computing the metrics, and subtask scores. 

Use the 
`config.yaml` file to set the model you want to use for the evaluation as in the training pipeline.
#### Example Usage:
Run the scripts with the following command:
```bash
python -m src.evaluate.predict --config config.yaml --dataname incidents_test_cleaned.csv
python -m src.evaluate.evaluate --config config.yaml
```


### Analysis
The `src/analysis/confusion_matrix.py` script generates the confusion matrix for the model predictions 
based on the majority and minority classes. 

The `src/analysis/find_missclassifications.py` script prints the misclassified samples based on the model predictions. 


The `src/analysis/shap_explainer.py` script generates the SHAP values given an index of a 
specific row in the dataset you want to explain.
The script has the following parameters:
* `category`: the category you want to explain (e.g. `hazard-category`)
* `dataname`: the name of the dataset you want to explain the model on and is on the `data/processed` folder.
* `index`: the index of the row you want to explain in the dataset.



For the three scripts, the $BERT_{base}$ and $BERT_{CW}$ is used by default based on the stored 
results in the `results/` folder, since they are used for the paper's error analysis.  

#### Example Usage:
Run the script with the following command:
```bash
python -m src.analysis.confusion_matrix
python -m src.analysis.find_missclassifications
python -m src.analysis.shap_explainer --category hazard-category --dataname incidents_test_cleaned.csv --index 25
```
### Hyperparameters Optimization using Optuna
The `src/finetune/finetune.py` script uses Optuna to optimize the hyperparameters of the model.

The script has the following parameters to run the optuna hyperparameter optimization.: 
* `field` (either `title` or `text`) 
* `category` (e.g. `hazard-category`) 
* `model_name`  (e.g. `bert`, `SVM`),
* `augmenter_name` (e.g. `contextual_words`),
* `n_trials` the number of trials



#### Example Usage:
Run the script with the following command:
```bash
python -m src.finetune.finetune --field text --category hazard-category --model_name bert --n_trials 10 --augmenter_name baseline
```
Run the hyperparameter optimization for all the models, categories and augmenters possible combinations.

### Jupyter Notebooks

The `notebooks` folder contains Jupyter notebooks that we used to run the Kruskal-Wallis test,
explore the data and the notebooks given by the organizers.


