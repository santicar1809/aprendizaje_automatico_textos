import pandas as pd
import numpy as np 
import tensorflow as tf
import os
import re
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.models.hyper_parameters import all_models
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import nltk
import spacy
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import transformers
import torch
import math
from tqdm.auto import tqdm

def clear_text(text):
    pattern=r"[^a-zA-Z']"
    text=re.sub(pattern," ",text)
    text=text.split()
    text=" ".join(text)
    text=text.lower()
    return text

def iterative_modeling(data):
    '''This function will bring the hyper parameters from all_model() 
    and wil create a complete report of the best model, estimator, 
    score and validation score'''
    
    models = all_models() 
    
    output_path = './files/modeling_output/model_fit/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results = []

    # Iterating the models
    models_name = ['dummie','cat','lgbm','lr','xg','rf']
    for model,i in zip(models,models_name):
        best_estimator, best_score, val_score = model_structure(data, model[1], model[2]) #data, pipeline, param_grid
        results.append([model[0],best_estimator,best_score, val_score])
        
        joblib.dump(best_estimator,output_path +f'best_random_{i}.joblib')
    results_df = pd.DataFrame(results, columns=['model','best_estimator','best_train_score','validation_score'])
    
    bert_results = bert_model(data) 
    #joblib.dump(tf_results[1],output_path +f'best_random_nn.joblib')
    bert_results[1].save(output_path+'best_random_nn.h5')
    # Concatening logistic models and neuronal network
    final_rev = pd.concat([results_df,bert_results[0]])
    final_rev.to_csv('./files/modeling_output/model_report.csv',index=False)

    return final_rev[['model','validation_score']]


def model_structure(data, pipeline, param_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    seed=12345
    
    data['review_norm']=data['review'].apply(clear_text)
    
    data_train=data[data['ds_part']=='train']
    data_test=data[data['ds_part']=='test']
    features_train=data_train['review_norm']
    features_test=data_test['review_norm']
    target_train=data_train['pos']
    target_test=data_test['pos']
    
    # Training the model
    gs = GridSearchCV(pipeline, param_grid, cv=2, scoring='roc_auc', n_jobs=-1, verbose=2)
    gs.fit(features_train,target_train)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    score_val = eval_model(best_estimator,features_test,target_test)
    print(f'AU-ROC: {score_val}')
    results = best_estimator, best_score, score_val 
    return results
    
def eval_model(model, train_features, train_target, test_features, test_target):

    eval_stats = {}

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):

        eval_stats[type] = {}

        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        

        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]

        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps

        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1')

        # ROC
        ax = axs[1]
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')
        ax.set_title(f'Curva ROC')

        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)

    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))

    print(df_eval_stats)

    return
## Network Model Structure

def BERT_text_to_embeddings(texts, max_length=512, batch_size=100, force_device=None, disable_progress_bar=False):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    config = transformers.BertConfig.from_pretrained('bert-base-uncased')
    model = transformers.BertModel.from_pretrained('bert-base-uncased')
    ids_list = []
    attention_mask_list = []

    # texto al id de relleno de tokens junto con sus máscaras de atención 
    for input_text in texts:
        ids = tokenizer.encode(input_text, add_special_tokens=True, truncation=True, max_length= max_length)
        
        padded =   np.array(ids + [0]*(max_length-len(ids)))
        attention_mask = np.where(padded != 0, 1, 0)
        
        ids_list.append(padded)
        attention_mask_list.append(attention_mask)
    
    if force_device is not None:
        device = torch.device(force_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.to(device)
    if not disable_progress_bar:
        print(f'Uso del dispositivo {device}.')
    
    # obtener insertados en lotes
    
    embeddings = []

    for i in tqdm(range(math.ceil(len(ids_list)/batch_size)), disable=disable_progress_bar):
            
        ids_batch = torch.LongTensor(ids_list[batch_size*i:batch_size*(i+1)]).to(device)
        # <escribe tu código aquí para crear attention_mask_batch
        attention_mask_batch = torch.LongTensor(attention_mask_list[batch_size*i:batch_size*(i+1)]).to(device)    
        with torch.no_grad():            
            model.eval()
            batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)   
        embeddings.append(batch_embeddings[0][:,0,:].detach().cpu().numpy())
        
    return np.concatenate(embeddings)

def bert_model(data):
    seed=12345
    features_new=data['review_norm'].apply(clear_text).sample(200,random_state=seed)
    target_new=data.loc[features_new.index,'pos']
    
    features_new = features_new.reset_index(drop=True)
    target_new = target_new.reset_index(drop=True)
    
    features_bert = BERT_text_to_embeddings(features_new)
    
    features_train,features_test,target_train,target_test=train_test_split(features_bert,target_new,test_size=0.5,random_state=12345)
    
    np.savez_compressed('features_bert.npz', train_features_bert=features_train, test_features_bert=features_test)

    # y cargar...
    with np.load('features_bert.npz') as data:
        train_features_bert = features_train
        test_features_bert = features_test
    
    model = LGBMClassifier(random_state=seed)
    model.fit(features_train,target_train)
    eval_model(model, features_train, target_train, features_test, target_test)
    
    # Evaluating the model
    y_pred = model.predict(features_test)
    auc_score = roc_auc_score(target_test, y_pred)
    print(f"AU-ROC Score: {auc_score}")
    results = ['AU-ROC Score',auc_score]
    results_df = pd.DataFrame({'model':[results[0]],'validation_score':[results[1]]})
    return results_df,model