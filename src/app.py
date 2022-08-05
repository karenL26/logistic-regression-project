import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# Downloading the data and copy it in a new data set
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', sep=';')
df=df_raw.copy()

# We remove the duplicated data
df = df[df.duplicated() == False]

# We replace the unknown values with the most frequent value
for col  in df.select_dtypes(include="O").columns:
    df[col] = np.where((df[col]=='unknown') , df[col].mode(), df[col])

# Remove the outliers for age
df=df.drop(df[df['age'] > 69.5].index)

# Remove the outliers for duration
df=df.drop(df[df['duration'] > 644.5].index)

# Remove the outliers for campaign
df=df.drop(df[df['campaign'] > 6.0].index)

# Remove the outliers for cons.conf.idx
df=df.drop(df[df['cons.conf.idx'] > -26.949999999999992].index)

# Removing irrelevant data
df.drop(['marital','contact', 'month','day_of_week'],axis=1, inplace=True)

# Converting age into categorical data by creating age-groups of ten years.
df['age_group'] = pd.cut(x=df['age'], bins=[10,20,30,40,50,60,70,80,90,100])

# Remove old age column
df.drop(['age'], axis=1, inplace=True)

# Inserting categories 'basic.9y','basic.6y','basic4y' into 'middle_school'
df['education'] = df['education'].replace({'basic.9y': 'middle_school', 'basic.6y': 'middle_school', 'basic.4y': 'middle_school'})

# Converting target variable into binary
df['y'] = df['y'].map({'no': 0, 'yes':1})

# Encoding anothers binary variables
df['default'] = df['default'].map({'no': 0, 'yes':1})
df['housing'] = df['housing'].map({'no': 0, 'yes':1})
df['loan'] = df['loan'].map({'no': 0, 'yes':1})

# Encoding nominal variables
df = pd.get_dummies(df,columns=['job', 'poutcome', 'age_group'], dtype='int64')

# Encoding ordinal variable
df['education']=df['education'].map({'illiterate': 0, 'middle_school':1, 'high.school':2, 'professional.course':3,'university.degree':4})

# Separating the target variable (y) from the predictors(X)
X=df.drop(['y'],axis=1) 
y=df['y'] 

# We use random Over-Sampling to add more copies to the minority class
os =  RandomOverSampler()
X_new,y_new=os.fit_resample(X,y)

# Spliting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building a Logistic Regression model with hyperparameters tune.
model = LogisticRegression(C= 1000, penalty='l2', solver= 'liblinear')
model.fit(X_train, y_train)

# We save the model with joblib
joblib_file = "lr_model.pkl"  
joblib.dump(model, joblib_file)