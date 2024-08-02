import pandas as pd
import os
import re
import numpy as np

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def columns_transformer(data):
    #Pasamos las columnas al modo snake_case
    columns=data.columns
    new_cols=[]
    for i in columns:
        i=to_snake_case(i)
        new_cols.append(i)
    data.columns=new_cols
    print(data.columns)
    return data

def preprocess_data(data):
    
    data = columns_transformer(data)
    columns=['runtime_minutes','genres']
    
    # Reemplazamos los valores '\N' por nulos
    
    for col in columns:
        data[col].replace('\\N',np.nan,inplace=True)
    
    # Pasamos la columna runtime_minutes a tipo numerico
    
    data['runtime_minutes']=pd.to_numeric(data['runtime_minutes'])
    
    # Imputamos los nulos
    
    nancols=['runtime_minutes','average_rating','votes']
    for column in nancols:
        data[column].fillna(data[column].mean(),inplace=True)
    
    data['genres'].fillna(data['genres'].mode()[0],inplace=True)
    
    path = './files/datasets/intermediate/'

    if not os.path.exists(path):
        os.makedirs(path)

    data.to_csv(path+'preprocessed_data.csv', index=False)

    print(f'Dataframe created at route: {path}preprocessed_data.csv ')

    return data