<img width="2560" height="1707" alt="image" src="https://github.com/user-attachments/assets/5b3c39bb-46d8-49ed-a00e-dc6dfda83e77" />



Rossmann Store Sales Prediction
Table of Contents

1. Introduction
This notebook aims to predict daily sales for Rossmann drug stores across several European countries. The challenge involves forecasting sales for 1,115 stores situated across Germany, France, and other parts of Europe. The goal is to develop a robust machine learning model that can accurately predict future sales based on historical sales data, promotional activities, store information, and date-related features.

2. Dataset
The dataset consists of three main files:

train.csv: Contains historical daily sales data for each store, including sales, customers, promotions, and holidays.
test.csv: Contains the test set, where sales predictions are required. This file has similar features to train.csv but lacks the Sales column.
store.csv: Provides supplementary information about each store, such as store type, assortment type, competition distance, and promotion details.
sample_submission.csv: A sample submission file in the correct format.

3. Data Loading and Inspection
   
The notebook begins by loading the train.csv, test.csv, and store.csv files into pandas DataFrames. Initial inspection of these DataFrames is performed to understand their structure, column types, and identify any immediate issues.

5. Data Merging
The train and test DataFrames are merged with the store DataFrame using the common Store ID column. This enriches the sales data with detailed store information, which is crucial for building a comprehensive predictive model.

6. Feature Engineering
Date Features
New features are extracted from the Date column to capture temporal patterns:

Year, Month, Day: Numerical representation of the date components.
WeekOfYear: The week number of the year.
Competition and Promotion Features
Additional features are engineered to quantify the impact of competition and promotions:

CompetetionOpen: Calculates the duration (in months) for which a competitor has been open. Missing values are filled, and negative values are handled.
Prom2Open: Calculates the duration (in months) for which Promo2 has been active. This feature accounts for the start year and week of Promo2.
IsPromoMonth: A binary feature indicating whether a particular month falls within the PromoInterval.
6. Data Preparation for Modeling
Before training the model, the data undergoes several preparation steps:

Handling Closed Stores: Rows where stores were closed (Open == 0) and consequently had no sales are removed from the training data, as these are not relevant for predicting active sales.
Defining Input and Target Columns: Features (input_cols) and the target variable (target_cols, i.e., Sales) are explicitly defined.
Handling Missing Values: CompetitionDistance is imputed with its maximum value.
Numerical Feature Scaling: Numerical features are scaled using MinMaxScaler to normalize their range, which can improve model performance.
Categorical Feature Encoding: Categorical features (StateHoliday, StoreType, Assortment) are converted into numerical representations using OneHotEncoder.
7. Model Training and Evaluation
An XGBoost Regressor model is chosen for its efficiency and performance in tabular data prediction tasks.

Initial Model
An initial model is trained with n_estimators = 20 and max_depth = 4 to establish a baseline RMSE. The sales distribution and decision trees are visualized.

Retrained Model (n_estimators=1000)
The model is retrained with an increased n_estimators = 1000 to observe the improvement in RMSE, demonstrating the impact of more estimators on accuracy.

8. Feature Importance Analysis
The importance of each feature in predicting sales is analyzed using model.feature_importances_. A bar plot visualizes the top 10 most influential features, providing insights into which factors drive sales the most.

9. K-Fold Cross-Validation
To ensure the model's robustness and generalize well to unseen data, K-Fold Cross-Validation (with 5 splits) is performed. The training and validation RMSE are recorded for each fold, and predictions are averaged across all models to provide a more stable estimate.

10. Hyperparameter Tuning
Key XGBoost hyperparameters are tuned to optimize model performance, using a train-validation split for efficiency:

n_estimators: Explored values from 100 to 5000.
max_depth: Explored values up to 10.
learning_rate: Explored values from 0.1 to 0.8.
The tuning process aims to find the best combination of parameters that minimizes the validation RMSE.

11. Final Model Training and Submission
After hyperparameter tuning, the final XGBoost model is trained using the identified optimal parameters (n_estimators = 2000, max_depth = 6, learning_rate = 0.25) on the full training dataset. Predictions are then made on the x_test DataFrame.

The submission.csv file is prepared by:

Populating the Sales column with the test_preds.
Ensuring sales predictions are non-negative.
Setting sales to 0 for stores that were closed on the test dates.
The final submission.csv is saved, ready for submission to the competition.

12. Saving the Model
The trained final_model along with the scaler and encoder objects, input_cols, target_cols, numeric_cols, categorical_cols, and max_distance are saved using joblib. This allows for easy loading and deployment of the complete prediction pipeline without needing to retrain the model.
