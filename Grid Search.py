
# coding: utf-8

# In[1]:


# Basic Imports 

import joblib
import os
import pandas as pd
import sklearn.metrics
import sys
import numpy as np
import sklearn.preprocessing as preprocessing

# Loading the Data 
# Cache the train and test data in {repo}/__data__.
cachedir = os.path.join(sys.path[0], '__data__')
memory = joblib.Memory(location=cachedir, verbose=0)

@memory.cache()
def get_data(subset='train'):
    '''
    this function downloads and returns the data as a pair of dependent and independent variables. 
    '''
    
    # Construct the data URL.
    csv_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/'
    csv_url += f'adult/adult.{"data" if subset == "train" else "test"}'
    # Define the column names.
    names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'earns_over_50K']
    # Read the CSV.
    print(f'Downloading {subset} dataset to __data__/ ...')
    df = pd.read_csv(
        csv_url,
        sep=', ',
        names=names,
        skiprows=int(subset == 'test'),
        na_values='?',engine='python')
    # Split into feature matrix X and labels y.
    df.earns_over_50K = df.earns_over_50K.str.contains('>').astype(int)
    X, y = df.drop(['earns_over_50K'], axis=1), df.earns_over_50K
    return X, y

def missing_values(X):
    '''
    this function takes in a dataframe and returns a cleaned version. 
    '''
    # convert to DF just to make sure 
    
    X_df = pd.DataFrame(X)
    
    # Deleting unnessary features 
    
    del X_df['education']
    
    # Filling missing data 
    
    X_df['workclass'].fillna(value='Private',inplace=True)
    X_df['occupation'].fillna(method='backfill',inplace=True)
    X_df['native-country'].fillna(value='United-States',inplace=True)
    
    
    return X_df

def encoding(df):
    '''
    this function performs label encoding on the given df 
    '''
    
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
            
    return result, encoders


# In[2]:


X_train, y_train = get_data()


# In[3]:


X_train,_ = encoding(missing_values(X_train))


# In[4]:


from sklearn.preprocessing import StandardScaler


# In[5]:


sc = StandardScaler()


# In[6]:


X_train = sc.fit_transform(X_train)


# In[7]:


print(X_train.shape)


# In[8]:


from sklearn.ensemble import GradientBoostingClassifier


# In[9]:


model = GradientBoostingClassifier()


# In[10]:


from sklearn.model_selection import GridSearchCV


# In[11]:


hyper_parameters = [{'loss': ['deviance','exponential']},{'learning_rate': [0.1,0.01,0.001]}, 
                   {'n_estimators': [100,125,150,175,200,225,250,275,300]},
                   {'min_samples_split': [2,3,4,5]}, 
                   {'max_depth': [3,4,5,6,7,8,9,10]}]


# In[12]:


grid_search = GridSearchCV(estimator = model,
                           param_grid = hyper_parameters,
                           scoring = 'accuracy',
                           cv = 10, n_jobs = -1)


# In[13]:


grid_search = grid_search.fit(X_train,y_train)


# In[14]:


print('Best Accuracy:',grid_search.best_score_)


# In[15]:


print('Best Model:')
print('\n')
print(grid_search.best_estimator_)


# In[16]:


print('Best Hyper-parameters:')
print('\n')
print(grid_search.best_params_)

