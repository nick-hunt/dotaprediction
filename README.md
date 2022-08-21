# Introduction
This repository contains the code used for my MSc Project: Predicting the outcome of Dota 2 matches given hero selection using graph neural networks.
Instructions are provided below for running different sections of the code.

# Repository description:
```bash
notebooks/
    data_acquisition.ipynb          Scripting to acquire the list of matches and respective match picks, and combining them
    dataset.py                      Defines DotaV1 and DotaV2, subclasses of the Spektral Dataset class, including initialisation methods
    exploratory.ipynb               Data quality checks and insights
    filtering.ipynb                 Creates and saves dataframe of standard filter, mmr group filters and duration group filters
    graph_data_creation.ipynb       Takes the combined matches/picks csv and generates DotaV1 and DotaV2 datasets, and scales features
    modelling.ipynb                 Models graph data using single team perspective
    modelling_2.ipynb               Models graph data using multi team perspective
    modelling_3.ipynb               Models match data using single team perspective and logistic regression
    plotting.ipynb                  Creates plots to be used in the report
.gitignore                          Contains file types and folders not to be tracked with Git
Pipfile                             Contains information for pipenv to create and maintain Python virtual environment
Pipfile.lock                        Contains information for pipenv to create and maintain Python virtual environment
README.md                           Readme markdown
```
# Instructions for use
## Clone repository
The address for this repository is:
https://github.com/nick-hunt/dotaprediction.git

Clone this onto your local machine. It is lightweight - it does not contain any data.
## Virtual environment
Open the now cloned local repository, and type "pipenv install" into the terminal. This will go through the process of creating a pipenv virtual environment with the necessary libraries and versions.
Ensure the virtual environment is activated before running any further code.
Alternatively, code can be run on the global environment providing all necessary packages are installed. Refer to the Pipfile for required libraries.
## Extracting Dota data

## Generate graph data

## Run graph neural network

Link to repo data:
https://drive.google.com/drive/folders/1a4KU4zgnDIfKRaa82IXpaDjN2WFEGH-l?usp=sharing
