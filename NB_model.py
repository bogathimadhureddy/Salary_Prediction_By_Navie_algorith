import os
os.chdir(r'C:\Users\madhu\OneDrive\Desktop\360 DigiTMG\DataScience\ASSIGNMENTS SOLVED BY ME\Naive Bayes') # changing the working directory.

os.getcwd()  # get the working directory

import pandas as pd # it is used for Data Manipulation.
import numpy as np # used for numerical calculation. 
from sklearn.feature_extraction.text import CountVectorizer # It is used to finds the frequency of words.

from imblearn.pipeline import make_pipeline ,Pipeline # it is used to make pipeline.

from sklearn.compose import ColumnTransformer # used for different transformations in to specific columns

from imblearn.over_sampling import SMOTE # It is used to balance the imbalanced data set.

from sklearn.naive_bayes import MultinomialNB # It is used to Naive Bayes model.

import matplotlib.pyplot as plt # Used for Data Visuvalization.

from sklearn.model_selection import train_test_split # It is used to split the data set into train set and test set.

from sklearn import metrics # metrics used to evalute the model.
from sklearn.impute import SimpleImputer # used for null values imputation.

import joblib # It is used to save the model.

import sweetviz # It is used for Auto EDA

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder,MinMaxScaler # it is use for preprocessing

from sklearn.preprocessing import FunctionTransformer # For defining the custom transform

from sqlalchemy import create_engine # used for database connection.

salary_test = pd.read_csv(r"C:\Users\madhu\OneDrive\Desktop\360 DigiTMG\DataScience\DATA SETS\Data Set\SalaryData_Test.csv") # reading the test data Frame

salary_train = pd.read_csv(r"C:\Users\madhu\OneDrive\Desktop\360 DigiTMG\DataScience\DATA SETS\Data Set\SalaryData_Train.csv") # reading the train data.

salary = salary_train.append(salary_test) # In future try with the concat command instead of append.

#engine = create_engine('mysql+pymysql://{}:{}@localhost/{}'.format('root','madhu123','salary_db')) # connecting to database

#salary_test.to_sql('salary_test_tbl',con = engine,chunksize=1000,if_exists = 'replace',index =False) # exporting the data to database

#salary_train.to_sql('salary_train_tbl',con = engine,chunksize=1000,if_exists = 'replace',index =False) # exporting the data to database 

#salary.to_sql('salary_tbl',con = engine,chunksize=1000,if_exists = 'replace',index =False) # exporting the data to database

salary.isna().sum() # checking the null values.

salary1 = salary.copy()

##############################################################
# Exploratory Data Analysis
report  = sweetviz.analyze(salary)

report.show_html('salary.html')

##############################################################
# Data Preprocessing 

# Education feature is dropping due to education no already present and 
#'native','workclass','race' are only one class present most and giving less relation with output variable.
  
salary.drop(['education','native','workclass','race'],inplace = True,axis =1)  

# The Below columns are less corelation with the Salary column.
salary.drop(['occupation','sex'],inplace = True,axis =1)

# Converting the all schooling class is taken as one class and remaining as like same increament. 

#salary['educationno'] = salary['educationno'].map({range(0,9): 1 ,9:2 ,10:3, 11:4, 12:5, 13:6, 14:7, 15:8, 16:9})
salary['educationno'].replace(range(1,9), 0, inplace = True)
salary['educationno'].replace(9, 1, inplace = True)
salary['educationno'].replace(10, 2, inplace = True)
salary['educationno'].replace(11, 3, inplace = True)
salary['educationno'].replace(12, 4, inplace = True)
salary['educationno'].replace(13, 5, inplace = True)
salary['educationno'].replace(14, 6, inplace = True)
salary['educationno'].replace(15, 7, inplace = True)
salary['educationno'].replace(16, 8, inplace = True)

salary.maritalstatus.value_counts() # finds the frequency of the maritalstatus of classes.

# ' Married-AF-spouse',' Married-spouse-absent' this represents that both are living separtely due to job or
# work issues but small difference is one is worked in armed force. so i replacing those values.

# ' Separated',' Divorced' both are nearly same only difference is one is not separated legally.
# They are separated because of distrabence between them.


salary['maritalstatus'] = np.where(salary['maritalstatus'] == ' Married-AF-spouse',' Married-spouse-absent', 
                                   np.where(salary['maritalstatus']==' Separated',' Divorced',salary['maritalstatus']))


salary.maritalstatus.value_counts() # finds the frequency of the maritalstatus of classes.

    
salary['educationno'] = salary['educationno'].astype('object') # Education number is a OrdinalType Data Type not numerical.

num_cols = salary.select_dtypes(include = ['int64']).columns # taking Numeric columns

salary['educationno'] = salary['educationno'].astype('int64') # No need to apply Ordinal Encoding because it is already it is the order.

nominal_cols = salary.select_dtypes(include = ['object']).columns[:2] # Taking nominal  Columns

# The Outliers are present in the Data set but only few observations only have above >0 so based on my knowledge i can't aplly outliers treatment. 

salary['Salary'] = LabelEncoder().fit_transform(salary.Salary) # the LabelEncoder is taken <=50K as 0 and >50K as 1


# Pipeline Creation for different Data types
# created a function for removing outliers
# log transformation is not possible beacause the value is present.
salary_train,salary_test = train_test_split(salary,test_size = 0.2,random_state = 0,stratify=(salary['Salary']))

def sqrt_trans(x):
    return np.power(x,1/30)

#pd.get_dummies(salary[nominal_cols],drop_first = True)
categ_pipeline = Pipeline([('Encoding',OneHotEncoder(drop = 'first'))])

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy = 'median')),
                         ('normalization',FunctionTransformer(func = sqrt_trans, validate = False)),
                         ('scaling',MinMaxScaler())
                         ])

# all this maked onepipeline by using ColumnTransformer
preprocess_pipeline = ColumnTransformer([('Continious', num_pipeline,num_cols),
                                         ('Categorical',categ_pipeline,nominal_cols)
                                       ],remainder = 'passthrough')

preprocessed = preprocess_pipeline.fit(salary_train)

joblib.dump(preprocessed,'processed1.pkl')

processed1 = joblib.load('processed1.pkl')

# Transformed the hole Data
salary_clean_train = pd.DataFrame(processed1.transform(salary_train))

salary_clean_test = pd.DataFrame(processed1.transform(salary_test))


nb = MultinomialNB()

nb.fit(salary_clean_train.iloc[:, :14],salary_clean_train.loc[:,14])

predicted = nb.predict(salary_clean_test.iloc[:, :14])

metrics.accuracy_score(predicted,salary_clean_test.loc[:,14])


predicted = nb.predict(salary_clean_train.iloc[:, :14])

metrics.accuracy_score(predicted,salary_clean_train.loc[:,14])

############################################
# Model Tuning - Hyperparameter optimization

nb = MultinomialNB(alpha=5)

nb.fit(salary_clean_train.iloc[:, :14],salary_clean_train.loc[:,14])

predicted = nb.predict(salary_clean_test.iloc[:, :14])

metrics.accuracy_score(predicted,salary_clean_test.loc[:,14])

#####################################################################
# Deploying the Best Model.
nb = MultinomialNB()

pipe1 = make_pipeline(preprocess_pipeline,nb)

pipe1 = pipe1.fit(salary_train.iloc[:,:7],salary_train['Salary'])

joblib.dump(pipe1,'pre_model.pkl')
model = joblib.load('pre_model.pkl')

predicted = model.predict(salary_test.iloc[:,:7])


metrics.accuracy_score(predicted,salary_test['Salary'])

predicted = pd.Series(predicted)
predicted = np.where(predicted == 0, '<=50K','>50K')

salary_test['Salary'] = predicted

print("accuracy: %.2f, sensitivity: %.2f, specificity: %.2f, precision: %.2f"  %
  (metrics.accuracy_score(salary_test['Salary'].ravel(), predicted),
  metrics.recall_score(salary_test['Salary'].ravel(), predicted),
  metrics.recall_score(salary_test['Salary'].ravel(), predicted, pos_label = 0),
  metrics.precision_score(salary_test['Salary'].ravel(), predicted)))

# Confusion Matrix - Heat Map
cm = metrics.confusion_matrix(salary_test['Salary'], predicted)
cmplot = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['<=50K', '>50K'])
cmplot.plot()
cmplot.ax_.set(title = 'Salary Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

