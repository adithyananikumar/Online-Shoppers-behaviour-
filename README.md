Online Shoppers Purchasing Intention Prediction
Machine Learning - Assignment 2
Name: Adithyan A Kumar  
BITS ID: 2025NS04038

1. Problem Statement
In this assignment, I have used machine learning classification algorithms to predict whether a customer will make a purchase from an online shopping website.
For this project, I selected the Online Shoppers Purchasing Intention Dataset from Kaggle. I implemented five classification models and compared their performance using different evaluation metrics.
The main objective is to understand how different machine learning algorithms perform on the same dataset and identify the model that gives better predictions.
2. Dataset Description
Dataset: Online Shoppers Purchasing Intention Dataset  
Source: Kaggle  
Dataset Link: https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset
The dataset contains 12,330 records and 18 columns. Out of these, 17 columns are input features and one column is the target variable.
The target variable is `Revenue`.
`Revenue = True`: Customer completed a purchase.
`Revenue = False`: Customer did not complete a purchase.
Some important features in the dataset are Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, Month, VisitorType and Weekend.
The dataset contains both numerical and categorical features.
Data Preprocessing
Before training the models, I performed the following steps:
Loaded the dataset using Pandas.
Separated the input features and target variable.
Converted the Revenue column into binary values (0 and 1).
Converted categorical variables into numerical format using OneHotEncoder.
Standardized numerical features using StandardScaler.
Divided the dataset into 80% training data and 20% testing data.
The training dataset contained 9,864 records, and the testing dataset contained 2,466 records.
3. GitHub Repository Link

https://github.com/adithyananikumar/Online-Shoppers-behaviour-
4. Machine Learning Models Used
I implemented the following five classification models:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Support Vector Machine (SVM)
All five models were trained using the same dataset.
Evaluation Metrics
I used six evaluation metrics to compare the performance of the models:
Accuracy: Measures the proportion of correct predictions.
AUC: Measures how well the model separates the two classes.
Precision: Measures how many predicted purchases were actually purchases.
Recall: Measures how many actual purchases were correctly identified.
F1 Score: Measures the balance between precision and recall.
MCC: Measures the overall quality of binary classification.
Model Comparison Table
The following results were obtained from my Streamlit application.
ML Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.8851	0.8991	0.7487	0.3873	0.5105	0.4845
Decision Tree	0.9717	0.9471	0.9062	0.9114	0.9088	0.8921
KNN	0.8975	0.9077	0.7771	0.4733	0.5883	0.5551
Naive Bayes	0.2875	0.7583	0.1767	0.9853	0.2997	0.1525
SVM	0.9056	0.9264	0.7814	0.5414	0.6396	0.6005
Note: These values were obtained by uploading the complete dataset of 12,330 records into Streamlit. Since some of these records were used during model training, these results should not be treated as independent test performance. The final evaluation should be repeated using only the held-out `test\_data.csv` containing 2,466 records.
5. Observations
Logistic Regression
Logistic Regression achieved an accuracy of 88.51% and an AUC score of 0.8991. However, the recall was only 0.3873, which means the model was unable to identify many customers who actually made purchases.
The F1 score was 0.5105. From these results, I observed that Logistic Regression was better at identifying customers who did not make purchases than those who did.
Decision Tree
Decision Tree achieved the highest accuracy of 97.17% among the five models in the current evaluation.
It also obtained a precision of 0.9062, recall of 0.9114, F1 score of 0.9088 and MCC of 0.8921.
These results show that the Decision Tree performed very well on the uploaded dataset. However, since the evaluation included training records, its performance may be affected by overfitting. It is necessary to verify the result using only the independent test data.
K-Nearest Neighbors (KNN)
KNN achieved an accuracy of 89.75% and an AUC score of 0.9077.
Its precision was 0.7771, but the recall was only 0.4733. This indicates that the model missed several customers who actually completed purchases.
Its F1 score was 0.5883, which was higher than Logistic Regression in this evaluation.
Gaussian Naive Bayes
Naive Bayes achieved the lowest accuracy of 28.75%.
However, it had the highest recall of 0.9853. This means that it identified almost all actual purchase cases, but it also incorrectly classified many non-purchasing sessions as purchases.
The precision was only 0.1767, and its F1 score was 0.2997.
From this result, I observed that Naive Bayes made a large number of false-positive predictions.
Support Vector Machine (SVM)
SVM achieved an accuracy of 90.56% and an AUC score of 0.9264.
It obtained a precision of 0.7814, recall of 0.5414, F1 score of 0.6396 and MCC of 0.6005.
The SVM performed better than Logistic Regression and KNN in terms of accuracy, F1 score and MCC in the current evaluation.
Overall Best Model
Based on the results obtained from the uploaded complete dataset, Decision Tree showed the highest accuracy, F1 score and MCC.
However, these results cannot be used to confirm the overall best model because the uploaded data included records used for training.
I need to evaluate all five models using only the independent test dataset before selecting the final winning model.
Final winner based on held-out test results: [Update after testing with test_data.csv]
6. Streamlit Application
I created a Streamlit web application to display and compare the performance of the five machine learning models.
The application includes:
CSV file upload option.
Dropdown menu for selecting machine learning models.
Display of Accuracy, AUC, Precision, Recall, F1 Score and MCC.
Confusion matrix.
Classification report.
Comparison table of all five models.
Sample predictions.
The application allows the user to select a model and upload test data to view the classification results.
Streamlit Application Link: [Add deployed Streamlit URL here]
7. Project Files
The GitHub repository should contain the following files:
`train\_models.py` - Code for training and evaluating the five classification models.
`app.py` - Streamlit web application.
`requirements.txt` - Python libraries required for the project.
`README.md` - Project description and results.
`test\_data.csv` - Dataset used for independent testing.
`metrics.csv` - Model evaluation results generated during training.
`models/` - Saved machine learning models.
8. How to Run the Project
First, install the required Python libraries:
```bash
pip install -r requirements.txt
```
Run the machine learning training script:
```bash
python train\_models.py
```
After training the models, execute the Streamlit application:
```bash
python -m streamlit run app.py
```
Upload the test dataset and select any of the five models to view its performance.
9. Conclusion
In this assignment, I implemented five machine learning classification algorithms using the Online Shoppers Purchasing Intention Dataset.
I compared their performance using Accuracy, AUC, Precision, Recall, F1 Score and MCC.
From the initial evaluation, Decision Tree obtained the highest scores for several metrics. Naive Bayes showed high recall but low precision, while SVM provided better F1 and MCC results than Logistic Regression and KNN.
This project helped me understand the importance of preprocessing, classification algorithms, evaluation metrics and deploying machine learning models through Streamlit.
The final model selection will be based on evaluation using the independent test dataset.
