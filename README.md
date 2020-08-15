![cover_photo](./LC-Logo-Official-min-1024x418.png)



## 1. Problem Identification

#### 1.1. Background: Lending Club

*[LendingClub](https://www.lendingclub.com/) is an American peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform. The company claims that $15.98 billion in loans had been originated through its platform up to December 31, 2015.

LendingClub enables borrowers to create unsecured personal loans between $1,000 and $40,000. The standard loan period is three years. Investors can search and browse the loan listings on LendingClub website and select loans that they want to invest in based on the information supplied about the borrower, amount of loan, loan grade, and loan purpose. Investors make money from interest. LendingClub makes money by charging borrowers an origination fee and investors a service fee.[Company Wiki Page](https://en.wikipedia.org/wiki/LendingClub)*

#### 1.2 Problem Statement

In this capstone project, my goal is to create an interest rate generator for unsecured personal loans issued by Lending Club based on the characteristics on each loan. 


## 2. Data Wrangling

#### 2.1. Data Collection

The dataset contains loan data for all loans issued through the 2007-2015, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. The file containing loan data through the "present" contains complete loan data for all loans issued through the previous completed calendar quarter. Additional features include credit scores, number of finance inquiries, address including zip codes, and state, and collections among others. The file is a matrix of about 890 thousand observations and 75 variables. The datast is acquired through the Kaggle API by clicking on the links below:


> * [Kaggle Dataset](https://www.kaggle.com/wendykan/lending-club-loan-data)

#### 2.2. Data Definition

I investigated the below features with the help of info(), describe(), and panda profiling. 

    1.	Column Name
    2.	Data Type (numeric, categorical, timestamp, etc)
    3.	Description of Column
    4.	Count or percent per unique values or codes (including NA)
    5.	The range of values or codes


## 3. Data Cleaning

* **Problem 1:** Handling missing data. **Solution:** use fillna() to imputate the missing value with its mean, median or mode. In some cases, just replace the missing data with zeros.

* **Problem 2:** Removing duplicates. **Solution:** use the built in Pandas DataFrame function drop_duplicates(). 


## 4. Exploratory Data Analysis

* **Boxplot:** Visualizing outliers
![Boxplot for Visulaizing Outlier](./Capstone2boxplot.png)

* **Heatmap:** Visualizing correlations
![Heatmap for Correlations](./Capstone2Pearson.png)
   

## 5. Pre-processing and Training Data Development

    •	Create dummy or indicator features for categorical variables
    •	Standardize the magnitude of numeric features: minmax or standard scaler
    •	Split into testing and training datasets


## 6. Modeling

#### 6.1. Method

There are three main types of regression models:

1. **Simple Regression:** Use for linear data.

2. **Random Forest Regressor:** Random Forest Regression. If the data is nonlinear, Ensemble Method generates better predictions.

3. **Random Forest Regressor XGboost:**

**WINNER:Random Forest Regressor XGboost** 

I chose Random Forest Regressor XGboost to accomodate the nonlinear nature of the dataset.

#### 6.2. Hyperparameter Tuning

###### 6.2.1. Grid Search Cross Validation

I applied GridSearchCV on Random Forest Regressor. Due to the time and resource constraints, I only varied one hyperparameter: n_estimators from 100 to 1100 at the interval of 100. In the end, the best parameter for n_estimators is 500. 

###### 6.2.2. Randomized Search Cross Validation

Here I was able to apply a set of parameters to search through. As a result of that, I implemented randomized search cross validation on Random Forest Regressor XGboost with the below parameter set:

  * 'colsample_bytree':[0.4, 0.6, 0.8]
  * 'gamma':[0, 0.03, 0.1, 0.3]
  * 'min_child_weight':[1.5, 6, 10]
  * 'learning_rate':[0.05, 0.1]
  * 'max_depth':[3,5,7]
  * 'n_estimators':[500]
  * 'reg_alpha':[1e-5, 1e-2,  0.75, 1]
  * 'reg_lambda':[1e-5, 1e-2, 0.45, 1, 1.5, 2]
  * 'subsample':[0.6, 0.95] 

#### 6.3. Model Evaluation Metrics

*MAE: mean of the absolute value of errors

*RMSE : squared root of the mean of the squared errors.

Method | RMSE
------------ | -------------
Random Forest Regressor| 1.3959306170693693
Random Forest Regressor XGB | 2.347896772001473
Random Forest Regressor XGB with Randomized Search CV | 1.281564628227814

>***NOTE:** I choose RMSE as the accuracy metric over mean absolute error(MAE) because the errors are squared before they are averaged which penalizes large errors more. Thus, the RMSE is more desirable when the large errors are unacceptable. The lower the RMSE, the better the prediction because the RMSE takes the square root of the residual errors of the line of best fit.*

## 7. Future Improvements

* SVM: use Kernel Trick along with other hyperparameter tuning to evaluate the accuracy of the model with respect to the aforementioned methods. 

* Bayesian Optimization: compare the efficacy and efficiency using Bayesian Optimization to Randomized Search Cross Validation.

* Due to RAM constraints, I had to train a 10% sample of the original dataset. Without resource limitations, I would pursue training on the full dataset. Preliminary tests showed that the bigger the training size, the lower the RMSE. 

## 8. Credits

Thanks to Jeremy Cunningham for being an amazing Springboard mentor.
