# Logistic Regression Project
This project used the folowing [data set](https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv')
In this [file](src/explore.ipynb) you can find a quick exploratory data analysis.
For this project we are trying to predict if the client has subscribed a term deposit?

## Cleaning Process
In the cleaning process we:
* Remove duplicated data
* Remove irrelevant data as 'marital','contact', 'month','day_of_week'.
* We remove outliers higher than the upper boundarie which is defined using the IQR method. The values removed correspond to age, duration, campaign, cons.conf.idx.
* For categorical values we replace the unknown values with the most frequent value
* Convertion of the data:
    * Age into categorical data by creating age-groups of ten years.
    * Education grouping 'basic.9y', 'basic.6y' and 'basic.4y' under 'middle_school'
    * Convert 'y', 'default' into binary values.
    * Convert 'default', 'housing', 'loan' into binary values.
    * Encoding nominal variables 'job', 'poutcome', 'age_group' into dummy variables.
    * Encoding ordinal variable 'education' into ordinal variable.

## Model
For the model we:
* Separating the target variable (y) from the predictors(X)
* In order to balance the target variable we use random Over-Sampling to add more copies to the minority class.
* Spliting the dataset into training and testing with the proportion 80%-20%.
* We perform a feature standar scaling.
* We finally building a Logistic Regression model with hyperparameters tune.