# Midterm Project - Heart Failure Detection

## Motivation

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide. Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

That's why in this project, we will try several machine learning classification models to come up with a solution. We'll try to predict DEATH_EVENT of a person based on the other features.

## Dataset

The dataset is collected from kaggel:  
https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

## Feature Description

- age: Person age.
- anaemia: Decrease of red blood cells or hemoglobin (boolean).
- creatinine_phosphokinase: Level of the CPK enzyme in the blood (mcg/L).
- diabetes: If the patient has diabetes (boolean).
- ejection_fraction: Percentage of blood leaving the heart at each contraction (percentage).
- high_blood_pressure: If the patient has hypertension (boolean).
- platelets: Platelets in the blood (kiloplatelets/mL).
- serum_creatinine: Level of serum creatinine in the blood (mg/dL).
- serum_sodium: Level of serum sodium in the blood (mEq/L).
- sex: Woman or man (binary).
- smoking: If the patient smokes or not (boolean).
- time: Follow-up period (days).
- DEATH_EVENT: If the patient deceased during the follow-up period (boolean).

## Repo Structure

Following files are included in the repo:

```
heart-failure-detection
├── Dockerfile <- Instructions to build Docker image
├── Pipfile <- Package dependency management file
├── Pipfile.lock <- Package dependency lock management file
├── README.md <- Getting started guide of the project
├── heart_failure_clinical_records_dataset.csv <- Dataset
├── model.bin <- Exported trained model
├── notebook.ipynb <- Jupyter notebook with all codes
├── predict.py <- Model prediction API script for local deployment
├── preduct_test.py <- Model prediction API script for testing
└── train.py <- Final model training script
```

## Installing Dependencies

For the project, Pipenv is used for package management. So, first we need to install Pipenv. Then run these commands to install the dependencies:

```
git clone <repo>
cd heart-failure-detection
pipenv install
```

## Running the Jupyter Notebook

Run jupyter notebook using the following command assuming we are inside the project directory:

```
jupyter notebook
```

## Running the Model Locally

The final model training codes are exported in this file. To train the model:

```
python train.py
``` 

For local deployment, start up the Flask server for prediction API:

```
python predict.py
```

It will run the server on localhost using port 8080.

Finally, send request to the prediction API `http://localhost:8080/predict` and get response:

```
python predict_test.py
```

## Running the Model in Cloud 

The model is deployed on **Pythonanywhere** and can be accessed using:

```
https://mdrkb.pythonanywhere.com/predict
```

The API takes JSON array of records as input and returns a response JSON array.

How to deploy a basic Flask application to Pythonanywhere can be found [here](https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-pythonanywhere.md). 
Only upload the `heart_failure_clinical_records_dataset.csv`, `train.py` and `predict.py` files inside app directory.
Then open a terminal and run `train.py` and `predict.py` files. Finally, reload the application.
If everything is okay, then the API should be up and running.

To test the cloud API, again run `predict_test.py` from locally using the cloud API URL. We'll get output similiar to this:

```
[
  {'heart_failure': True, 
    'heart_failure_probability': '99.99%', 
    'id': 0
  }, 
  {
    'heart_failure': False, 
    'heart_failure_probability': '99.886%', 
    'id': 1
  }
]
```

Notes:
- Pythonanywhere instance has almost all the necessary packages installed. So, we don't need to install packages manually.
- Some packages are not up to date. So, there might be compatibility issue specially with Scikit-Learn and XGBoost.
For example changing `dv.get_feature_names_out()` to `dv.get_feature_names()` for lower versions of Scikit-Learn.

## Running the Docker Image

Will be added soon.
