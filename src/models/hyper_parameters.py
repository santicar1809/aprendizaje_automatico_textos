import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
import logging
import math
from tqdm.auto import tqdm
import nltk
import spacy
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import transformers
import torch
import math
from tqdm.auto import tqdm

class NLTKTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [' '.join(self._lemmatize(doc)) for doc in X]
    
    def _lemmatize(self, text):
        tokens = word_tokenize(text)
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmas
    
def spacy_lemma(data):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    doc=nlp(data)
    lemmas=[token.lemma_ for token in doc]
    return " ".join(lemmas)

## Logistic Regression Model
def all_models():
    '''This function will host all the model parameters, can be used to iterate the
    grid search '''

    # 
    dummie_pipeline = Pipeline([
        ('NLKT',NLTKTokenizer()),
        ('TF', TfidfVectorizer(stop_words=list(nltk_stopwords.words('english')))),
        ('dummie',DummyClassifier(strategy="constant",constant=1))
    ])

    dummie_param_grid = {
            'dummie__strategy': ['constant'],  # Regularización
            'dummie__constant': [1]  # Fuerza de la regularización
            }

    dummie = ['dummie',dummie_pipeline,dummie_param_grid]
    
    cat_param_grid = {
        'cat__iterations': range(50, 201, 50),
        'cat__depth': range(1, 11)
    }
    
    cat_pipeline = Pipeline([
    ('NLKT',NLTKTokenizer()),
    ('TF', TfidfVectorizer(stop_words=list(nltk_stopwords.words('english')))),
    ('cat',CatBoostClassifier(random_state=1234))])
    
    cat = ['cat',cat_pipeline,cat_param_grid]
    
    lgbm_pipeline = Pipeline([
        ('TF', TfidfVectorizer(tokenizer=spacy_lemma)),
        ('lightgbm',LGBMClassifier())
    ])
    
    lgbm_param_grid = {
        'lightgbm__max_depth': [3, 5, 7],  # Profundidad máxima del árbol
        'lightgbm__learning_rate': [0.1, 0.01, 0.001],  # Tasa de aprendizaje
        'lightgbm__n_estimators': [100, 500, 1000],  # Número de árboles en el bosque
        
    }
    lgbm = ['lightgbm',lgbm_pipeline,lgbm_param_grid]
    
    lr_pipeline = Pipeline([
        ('NLKT',NLTKTokenizer()),
        ('TF', TfidfVectorizer(stop_words=list(nltk_stopwords.words('english')))),
        ('logreg', LogisticRegression(max_iter=10000))
    ])
    
    lr_param_grid = {
        'logreg__penalty': [ 'l1', 'l2', None],  # Regularización
        'logreg__C': [0.01, 0.1, 1, 10, 100],  # Fuerza de la regularización
        'logreg__solver': ['saga'], # ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algoritmo de optimización
        #'logreg__l1_ratio': np.linspace(0, 1, 10)  # Solo si el solver es 'saga' y penalty es 'elasticnet'
    }

    lr = ['logreg',lr_pipeline,lr_param_grid]
    
    xg_pipeline=Pipeline([
        ('NLKT',NLTKTokenizer()),
        ('TF', TfidfVectorizer(stop_words=list(nltk_stopwords.words('english')))),
        ('xgboost',XGBClassifier(random_state=1234))
    ])
    
    xg_param_grid = {
        'xgboost__max_depth': [3, 5, 7],  # Profundidad máxima del árbol
        'xgboost__learning_rate': [0.1, 0.01, 0.001],  # Tasa de aprendizaje
        'xgboost__n_estimators': [100, 500, 1000]  # Número de árboles en el bosque
    }

    xg = ['xgboost',xg_pipeline,xg_param_grid]
    
    rf_pipeline = Pipeline([
    ('NLKT',NLTKTokenizer()),
    ('TF', TfidfVectorizer(stop_words=list(nltk_stopwords.words('english')))),
    ('random_forest', RandomForestClassifier(random_state=1234))])

    # Crear el grid de parámetros para Random Forest
    rf_param_grid = {
        'random_forest__n_estimators': [100, 500, 1000],  # Número de árboles en el bosque
        'random_forest__max_depth': [10, 20, 30],  # Profundidad máxima del árbol
        'random_forest__min_samples_split': [2, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno
        'random_forest__min_samples_leaf': [1, 2, 4],  # Número mínimo de muestras requeridas para estar en un nodo hoja
    }

    # Evaluar el modelo con la función model_evaluation
    rf = ['random_forest',rf_pipeline,rf_param_grid]
    
    models = [dummie,cat,lgbm,lr,xg,rf] #Activate to run all the models
    return models