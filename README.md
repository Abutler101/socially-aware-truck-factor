# Replication Package - Measuring Software Resilience Using Socially Aware Truck Factor Estimation
### Alexis Butler, Dan O'Keeffe, Santanu Kumar Dash

## Contents
- **./data_collection** → Script to collate VCS data for the target projects
- **./analysis** → Script to run baseline and proposed estimators against the full dataset
- **./post_processing** → Script to clean estimator output - catch contributor duplication that arises from alises missed by automated alias grouping
- **./indepth_eval** → Script to evaluate the performance of the estimators
- **./shared_models** → Collection of datamodels used by the various scripts

## Requirements
- Python 3.8
- A Python Virtual Environment manager (conda etc.)

## Setup
### Dataset
- Download the zipped dataset from Zenodo: https://zenodo.org/records/15223467
- Un-zip the dataset (result will contain a folder called 'output' and a json file `truck_factors.json`)
- move the folder `output` into data_collection
- move the json file `truck_factors.json` to the root of this repo
### Scripts
- Create a Python3.8 virtual environment
- Install dependencies from requirements.txt

## Usage Notes
- Scripts are inter-dependant:
  - `data_collection.main` → `analysis.main` → `post_processing.main` → `indepth_eval.main`
- assuming dataset is downloaded execution can start from `analysis.main`
- post_processing and indepth_eval scripts require human input as directed in the console

## Contact
Please raise any issues or questions using the built-in GitHub Issue system, Alexis will address them in due course.

## Paper
```
Raw Bibtex cite to paper - TBC
```