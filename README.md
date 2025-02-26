# Heart Failure Prediction and Analysis
## Introduction
I worked on a Heart Failure Prediction and Analysis project, where the goal was to build a machine learning model that could predict individuals at risk of heart failure based on key medical attributes. Cardiovascular diseases are one of the leading causes of death worldwide, and early detection plays a crucial role in preventing fatalities.
To achieve this, I used R programming for data preprocessing, statistical modeling, and visualization. The dataset contained key medical parameters such as Age, Sex, RestingBP, Cholesterol, ExerciseAngina, and ST_Slope, which were analyzed to develop an effective classification model.
________________________________________
## Project Overview
In this project, I aimed to:
•	Develop a classification model that predicts the likelihood of heart failure.

•	Analyze medical attributes and their correlation with heart disease.

•	Generate insights that can help in early detection and intervention.

To achieve this, I used two classification models:
1.	Logistic Regression – A widely used model in healthcare predictions.
2.	Decision Tree – To identify key contributing factors to heart failure.
________________________________________
## Understanding the Problem
### Why This Project?
•	Heart diseases cause 17.9 million deaths globally each year (WHO).

•	Healthcare professionals need predictive models to detect risks early.

•	Machine learning can help identify patterns in patient data that might not be evident through traditional analysis.

## Business Understanding
The primary goal of this project was to help medical professionals and hospitals make data-driven decisions. By predicting heart failure risk in patients, healthcare providers can:
•	Offer early medical interventions and lifestyle changes.

•	Reduce hospital readmissions and long-term care costs.

•	Optimize resource allocation for high-risk patients.
________________________________________
## Working with the Data
### Step 1: Data Collection & Cleaning
I started by importing the dataset into R and checking for data quality issues:
•	Removed duplicates and missing values.
•	Handled outliers (e.g., extreme cholesterol values).
•	Converted categorical variables (e.g., Chest Pain Type) into numerical representations.
The dataset contained 12 medical features relevant to heart disease.

### Step 2: Feature Selection
Analyzed the dataset to determine which features were most important for prediction:
•	Age & RestingBP showed strong correlation with heart disease.

•	ExerciseAngina & ST_Slope played a significant role in heart failure risk.

•	Cholesterol levels needed outlier handling before model training.
________________________________________
## Model Development & Evaluation
### Step 3: Model Selection
Since this was a binary classification problem, I used:
1.	**Logistic Regression** – Simple yet powerful in healthcare predictions.
2.	**Decision Tree** – To help visualize which features contributed the most.
   
### Step 4: Model Evaluation
I evaluated both models based on:
•	Accuracy (Correct Predictions / Total Predictions)

•	Sensitivity (How well the model detects actual heart disease cases)

•	Specificity (How well the model avoids false positives)

Model	             Accuracy	Sensitivity	Specificity
Logistic Regression	 88.00%	  91.45%	     83.74%
Decision Tree	       85.09%	  93.42%	     74.80%
The Logistic Regression model performed best with an 88% accuracy and was chosen as the final model.
________________________________________
## Key Insights from the Project
1.	Age and Blood Pressure were the strongest indicators of heart failure risk.
2.	Exercise Angina (chest pain during physical activity) was a significant predictor.
3.	ST_Slope variations helped in identifying patients with higher risks.
4.	Cholesterol levels alone were not sufficient to predict heart failure without additional factors.
________________________________________
## Challenges Faced
•	Handling Missing Data: Some patients had incomplete medical records.

•	Avoiding Overfitting: The Decision Tree model initially performed well but was too specific to training data.

•	Feature Selection: Ensuring the right balance of features without adding noise to the model.
________________________________________
## Future Enhancements
•	Adding More Data: Using larger, more diverse datasets for better generalization.

•	Using Deep Learning Models: Exploring Neural Networks for more complex predictions.
•	Explainable AI: Implementing SHAP values to understand model decision-making better.
________________________________________
## Conclusion
This project was an exciting opportunity to apply machine learning in the healthcare domain. By analyzing medical features and developing predictive models, I was able to provide valuable insights into heart failure risks. This work could be expanded further by integrating real-world patient data to improve medical decision-making.
By using data-driven approaches, hospitals and healthcare professionals can better manage preventative care, ultimately saving lives through early detection.






