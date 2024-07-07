## 2401FTDS_Classification_Project

# Analysing News Articles Dataset


![](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](URL_TO_YOUR_APP)

<div id="main image" align="center">
  <img src="https://github.com/ereshia/2401FTDS_Classification_Project/blob/main/announcement-article-articles-copy-coverage.jpg" width="550" height="300" alt=""/>
</div>

## Table of contents
* [1. Introduction](#Introduction)
* [2. Data Loading and Exploration](#Data-Loading-and-Exploration)
* [3. Data Preprocessing](#Data-Preprocessing)
* [4. Model Building](#Model-Building)
* [5. Evaluation](#Evaluation)
* [6. MLflow Integration](#MLflow)
* [7. Team Members](#team-members)


## 1. Introduction <a class="anchor" id="Introduction"></a>
This notebook contains a comprehensive analysis for a classification problem. It involves data loading, preprocessing, model building, and evaluation to predict the target variable based on given features.

## 2. Data Loading and Exploration <a class="anchor" id="Data-Loading-and-Exploration"></a>
In this section, we load the dataset and perform initial exploration to understand the structure and content of the data. Key steps include:
- Importing necessary libraries.
- Loading the dataset using pandas.
- Displaying the first few rows of the dataset.
- Checking for missing values and basic statistics of the dataset.


## 3. Data Preprocessing <a class="anchor" id="Data-Preprocessing"></a>
Data preprocessing involves cleaning and preparing the data for model building. Steps include:
- Handling missing values.
- Encoding categorical variables.
- Feature scaling if necessary.
- Splitting the dataset into training and test sets.

## 4. Model Building <a class="anchor" id="Model-Building"></a>
Here, we build and train various machine learning models to classify the target variable. This section covers:
- Selecting appropriate machine learning algorithms (e.g., Logistic Regression, KNN, SVM, Random Forest, etc.).
- Training the models on the training set.
- Hyperparameter tuning to improve model performance.
 

## 5. Evaluation <a class="anchor" id="Evaluation"></a>
After building the models, we evaluate their performance using various metrics. This section includes:
- Making predictions on the test set.
- Calculating accuracy, precision, recall, F1-score, and other relevant metrics.
- Comparing the performance of different models.
- Summarizing the findings and providing recommendations or next steps.


## 6. MLflow Integration <a class="anchor" id="MLflow"></a>
MLflow is used to track and manage experiments, models, and metrics throughout the machine learning lifecycle. This section includes:

Setting up MLflow.
* Logging experiments and parameters.
* Tracking model performance metrics.
* Saving and loading models using MLflow.

# Steps to Use MLflow
* Install MLflow:

  pip install mlflow
* Set Up MLflow Tracking Server:
  
  mlflow server
* Log Parameters and Metrics:
  
  import mlflow
  
  import mlflow.sklearn

  mlflow.start_run()
# Log parameters
  mlflow.log_param("param_name", param_value)
# Log metrics
  mlflow.log_metric("metric_name", metric_value)
# Log model
  mlflow.sklearn.log_model(model, "model")

  mlflow.end_run()
* View Results:
Access the MLflow UI to view logged experiments and models at http://localhost:5000.


## 7. Team Members<a class="anchor" id="team-members"></a>

| Name                                                                                        |  Email              
|---------------------------------------------------------------------------------------------|--------------------             
| [Refilwe Masupu](https://github.com/Refilwemasapu)                                          | skymasapu12@gmail.com
| [Sihle Kalolo](https://github.com/Toni8)                                                    | kalolohlesi@gmail.com
| [Ishmael Ngobeni](https://github.com/thatsso-ish)                                           | ismaelkatlego0@gmail.com
| [Cleragy Jilani](https://github.com/cleragy)                                                | kanunimkonza@gmail.com
| [Msizi Mzobe](https://github.com/OMMzobe)                                                   | ommzobe@gmail.com
| [Sandile Jali](https://github.com/sandilejali)                                              | sandilejali31@gmail.com
