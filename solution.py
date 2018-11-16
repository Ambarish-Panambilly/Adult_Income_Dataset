
# coding: utf-8

# In[1]:


def get_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    
    
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('gbc',  GradientBoostingClassifier(max_depth = 5)))
    model = Pipeline(estimators)
    return model

