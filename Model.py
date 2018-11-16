
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

def score_solution():
    '''
    this function fits the pipeline to training data and tests the model on the testing data and finally returns roc_auc_score 
    '''
    
    import solution
    pipeline = solution.get_pipeline()
    
    error_message = 'Your `solution.get_pipeline` implementation should '         'return an `sklearn.pipeline.Pipeline`.'
    assert isinstance(pipeline, sklearn.pipeline.Pipeline), error_message
    
    # Train the model on the training DataFrame.
    
    X_train, y_train = get_data(subset='train')
    X_train,_ = encoding(missing_values(X_train))
    print('\n')
    print('Training........')
    print('\n')
    pipeline.fit(X_train, y_train)
    
    # Apply the model to the test DataFrame. 
    
    X_test, y_test = get_data(subset='test')
    X_test,_ = encoding(missing_values(X_test))
    print('\n')
    print('Test Results')
    print('\n')
    y_pred = pipeline.predict_proba(X_test)
    
    assert (y_pred.ndim == 1) or         (y_pred.ndim == 2 and y_pred.shape[1] == 2),         'The predicted probabilities should match sklearn''s '         '`predict_proba` output shape.'
    y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, 1]
    
    return sklearn.metrics.roc_auc_score(y_test, y_pred)

if __name__ == '__main__':
    print('roc_auc_score:',score_solution())

