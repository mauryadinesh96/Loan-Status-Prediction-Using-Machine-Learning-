# Loan-Status-Prediction-Using-Machine-Learning
# Classification Loan Status


# Introduction:
we are going to work on binary classification problem, where we got some information about sample of peoples , and we need to predict whether we should give some one a loan or not depending on his information . we actually have a few sample size (614 rows), so we will go with machine learning techniques to solve our problem .
# About the data
Loan_ID : Unique Loan ID

Gender : Male/ Female

Married : Applicant married (Y/N)

Dependents : Number of dependents

Education : Applicant Education (Graduate/ Under Graduate)

Self_Employed : Self employed (Y/N)

ApplicantIncome : Applicant income

CoapplicantIncome : Coapplicant income

LoanAmount : Loan amount in thousands of dollars

Loan_Amount_Term : Term of loan in months

Credit_History : Credit history meets guidelines yes or no

Property_Area : Urban/ Semi Urban/ Rural

Loan_Status : Loan approved (Y/N) this is the target variable
## Steps Involved
 
 ### Dataset
 
The foremost thing that is needed the most, in a machine learning project, is a dataset.
A dataset, as the name says, is a set or collection of data. It contains all the values that we need to work with, to get the necessary results.
For this project, the dataset that I have used is the one I found on Kaggle. You can use any dataset available on the internet that you feel comfortable working with.

### Importing Dependencies : 
The dependencies that we will be using are :
numpy, pandas, seaborn, and ScikitLearn.

### Data Collection and Processing

![image](https://user-images.githubusercontent.com/108235140/203012007-cb22d5e5-af49-479d-bff0-28653c3c8c6b.png)

- The dataset will be in the format of a CSV. Thus to read it, I am taking the help of the pandas method, called read_csv()
- Also I am storing the dataset in the variable called “loan_dataset” to refer to the entire dataset by this variable name.
- Now, I have checked the number of rows and columns in the dataset.
- After that I have checked, how many values are missing from the dataset.
- Furthur, I have deal with missing values and remove that anomaly using various methods.
- Now, After that, I have performed Label Encoding. I am replacing the loan status of ‘Y’ with 1, and ‘N’ with 0, for better reference.
- Then I have count the frequency of each value in the “Dependent” column. Replace the value of 3+ with 4 to better performance.
- Again, I convert the categorical columns, to numerical values for better reference.
- And lastly in this step, split the data and label into the X and Y variables : X will store all the features on which the loan status depends, excluding the loan status itself. Y will store only the Loan Statuses.

### Splitting X and Y into Training and Testing Variables

Here, I am splitting the data into four variables, viz., X_train, Y_train, X_test, Y_test. The testsize represents the ratio of how the data is distributed among X_trai and X_test (Here 0.2 means that the data will be segregated in the X_train and X_test variables in an 80:20 ratio). You can use any value you want. A value < 0.3 is preferred.


# what you will learn in this kernel ?

basics of visualizing the data .

# how to compare between feature importance (at less in this data) .

feature selection

feature engineer

# some simple techniques to process the data .

handling missing data .

how to deal with categorical and numerical data .

outliers data detection

but the most important thing that you will learn , is how to evaluate your model at every step you take .

# what we will use ?
some important libraries like sklearn, matplotlib, numpy, pandas, seaborn, scipy

fill the values using backward 'bfill' method for numerical columns , and most frequent value for categorical columns (simple techniques)

## 5 different models to train your data, so we can compare between them
1.Logistics Regression

2.Support Vector Classifiers (SVC)

3.Decision Tree Classifier

4.Random Forest Classifier

5.Gradient Boosting Classifier

To predict the accuracy we will use the accuracy score function from scikit-learn library.




## A brief about Support Vector Machine Model

The algorithm that I am using for this purpose, is the Support Vector Machine. Support Vector Machine,(SVM), falls under the “supervised machine learning algorithms” category. It can be used for classification, as well as for regression. In this model, I plot each data item as a unique point in an n-dimension,(where n is actually, the number of features that we have), with the value of each of the features being the value of that particular coordinate. Then, I perform the process of classification by finding the hyper-plane that differentiates the two classes.

## Why choose Support Vector Machine over other algorithms?

It all depends on the type of operations we are performing, and the type of data we are dealing with. SVM is preferred over other algorithms when :
- The data is not regularly distributed.
- SVM is generally known to not suffer the condition of overfitting.
- Performance of SVM, and its generalization is better on the dataset.
- And, lastly, SVM is known to have the best results for classification types of problems.

![image](https://user-images.githubusercontent.com/108235140/203007501-95bd2f23-0c4f-4daa-be8a-7960b675c2a8.png)
