[Description](#Description)     |       [Installation](#Installation)    |       [Usage](#Usage)    |       [Visuals](#Visuals)     | [Contributors](#Contributors)    |      [Timeline](#Timeline)       |       [List of Improvements](#list-of-improvements)  

## **Description**

Real estate Machine Learning challenge of the BeCode AI Bootcamp

The real estate company "ImmoEliza" wants to establish itself as the biggest one in all of Belgium. To pursue this goal, it needs to create a machine learning model to predict prices on Belgium's sales. That way, they can pick out the properties that are the most valuable to them.

This is my Machine Learning Model using Neural Network solution with Tensorflow and Keras.

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/Dadelidodu/immo_ml
2. Navigate into the project directory:
   ```bash
   cd immo_ml

3. Install the required Python packages
   ```bash
   pip install -r requirements.txt

4. Make sure the necessary files are available in '/data' directory.

## **Usage**
1. Run the main.py script:
   ```bash
   output is 
    data/cleaned_dataset.csv
    data/X_test.csv
    data/X_train.csv
    data/y_test.csv
    data/y_train.csv

2. Results of training:
   ```bash

    Training Metrics

    MAE: *71125.8671875*
    RMSE: *133836.14384220913*
    R²: *0.8269559144973755*
    MAPE: *15.500440206535332*
    sMAPE: *15.261694863585006*

    Testing Metrics

    MAE: *75700.5703125*
    RMSE: *128360.92187814333*
    R²: *0.8356913328170776*
    MAPE: *17.02116282618461*
    sMAPE: *16.900477202572294*
   

3. Results of SHAP Analysis:
   ```bash
   Mean Revenue and Mean Price per Locality clearly have a major impact on prediction.

## **Visuals**
Visual representations are crucial for understanding data trends and patterns. Key visualizations created in this project include:

- Training & Test Loss Comparison : https://ibb.co/Jycsjs1
- Heat Map : https://ibb.co/kG0gG47
   
## **Contributors**
David - https://github.com/Dadelidodu

## **Timeline**
Nov 2024 - project initiated at BeCode Brussels AI & Data Science Bootcamp

Dec 2024 - project concluded

## **List of Improvements**
Future enhancements for the project may include:

- Incorporating additional datasets from Statbel for more comprehensive insights
- Exploring Tuning of Neural Network Parameters for better prediction results
