from catboost import CatBoostClassifier
import pandas as pd

model1 = CatBoostClassifier()
model1.load_model('income_model',format= 'cbm')
varlist = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',  'maritalstatus', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hrs_per_week', 'native_country']

def get_incomelevel(dataset):
    '''
    inputs: dataset with transactions for which category needs to be assigned
    outputs: dataset with the additional column of predicted category
    '''
    #resetting the dataset to avoid any non standard indices
    dataset = dataset.reset_index(drop = True)
    #subsetting the dataset to only those columns that the model requires for scoring.
    data_subcols = dataset[varlist]
    predict = model1.predict(data_subcols)
    dataset['Incogrp'] = predict
    
    return dataset