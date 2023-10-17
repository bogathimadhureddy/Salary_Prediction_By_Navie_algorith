from flask import Flask, render_template, request
import re
import pandas as pd
import numpy as np
import copy
import joblib
from sqlalchemy import create_engine
  
# creating Engine which connect to postgreSQL
engine = create_engine('mysql+pymysql://{}:{}@localhost/{}'.format('root','madhu123','salary_db'))

def sqrt_trans(x):
    return np.power(x,1/30)
# Load the saved model
model = joblib.load('pre_model.pkl')

# Define flask
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        try:

            salary_data = pd.read_csv(f)
        except:
                try:
                    salary_data = pd.read_excel(f)
                except:      
                    salary_data = pd.DataFrame(f)

        salary_data = salary_data.iloc[:,:13] # When new data comes for that data there is no output varaible so no need this line.
        salary = salary_data.copy()

        salary_data.drop(['education','native','workclass','race'],inplace = True,axis =1) 

        salary_data.drop(['occupation','sex'],inplace = True,axis =1)

        salary_data['educationno'].replace(range(1,9), 0, inplace = True)
        salary_data['educationno'].replace(9, 1, inplace = True)
        salary_data['educationno'].replace(10, 2, inplace = True)
        salary_data['educationno'].replace(11, 3, inplace = True)
        salary_data['educationno'].replace(12, 4, inplace = True)
        salary_data['educationno'].replace(13, 5, inplace = True)
        salary_data['educationno'].replace(14, 6, inplace = True)
        salary_data['educationno'].replace(15, 7, inplace = True)
        salary_data['educationno'].replace(16, 8, inplace = True)   

        salary_data['maritalstatus'] = np.where(salary_data['maritalstatus'] == ' Married-AF-spouse',' Married-spouse-absent', 
                                   np.where(salary_data['maritalstatus']==' Separated',' Divorced',salary_data['maritalstatus']))

        salary_data['educationno'] = salary_data['educationno'].astype('object') # Education number is a OrdinalType Data Type not numerical.

        num_cols = salary_data.select_dtypes(include = ['int64']).columns # taking Numeric columns

        salary_data['educationno'] = salary_data['educationno'].astype('int64') # No need to apply Ordinal Encoding because it is already it is the order.

        nominal_cols = salary_data.select_dtypes(include = ['object']).columns # Taking nominal  Columns

        # The Outliers are present in the Data set but only few observations only have above >0 so based on my knowledge i can't aplly outliers treatment.                             

        model = joblib.load('pre_model.pkl')
            
        test_pred_lap = pd.DataFrame(model.predict(salary_data))
        test_pred_lap.columns = ["Salary"]
        predicted = pd.DataFrame(np.where(test_pred_lap['Salary'] == 0, '<=50K','>50K'))
        predicted.columns = ["Salary"]
        final = pd.concat([salary, predicted], axis = 1)
        

        final.to_sql('salary_predict', con = engine, if_exists = 'replace', index= False)
        
               
        return render_template("new.html", Y = final.to_html(justify = 'center'))

if __name__=='__main__':
    app.run(debug = True)
