import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
from src.models.built_models import BERT_text_to_embeddings
import re
import joblib
import transformers
import torch
import math
from tqdm.auto import tqdm

my_reviews = pd.DataFrame([
    'I did not simply like it, not my kind of movie.',
    'Well, I was bored and felt asleep in the middle of the movie.',
    'I was really fascinated with the movie',    
    'Even the actors looked really old and disinterested, and they got paid to be in the movie. What a soulless cash grab.',
    'I didn\'t expect the reboot to be so good! Writers really cared about the source material',
    'The movie had its upsides and downsides, but I feel like overall it\'s a decent flick. I could see myself going to see it again.',
    'What a rotten attempt at a comedy. Not a single joke lands, everyone acts annoying and loud, even kids won\'t like this!',
    'Launching on Netflix was a brave move & I really appreciate being able to binge on episode after episode, of this exciting intelligent new drama.'
], columns=['review'])

def clear_text(text):
    pattern=r"[^a-zA-Z']"
    text=re.sub(pattern," ",text)
    text=text.split()
    text=" ".join(text)
    text=text.lower()
    return text

def test_main():
    my_reviews_fit = my_reviews['review'].apply(clear_text)
    models_name=['dummie','cat','lgbm','lr','xg','rf']
    
    for model in models_name:    
        new_model = joblib.load(f'./files/modeling_output/model_fit/best_random_{model}.joblib')
        my_reviews_pred_prob = new_model.predict_proba(my_reviews_fit)[:, 1]

        result=[]
        for i, review in enumerate(my_reviews_fit.str.slice(0, 100)):
            print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')
            result.append(f'{my_reviews_pred_prob[i]:.2f}:  {review}')
        result_df=pd.Series(result)
        result_df.to_csv(f'./test/results/result_{model}.csv',index=False,header=False)
    
    my_reviews_bert = BERT_text_to_embeddings(my_reviews_fit)
    model_bert=joblib.load(f'./files/modeling_output/model_fit/best_random_bert.joblib')
    my_reviews_pred_prob = model_bert.predict_proba(my_reviews_bert)[:, 1]
    result=[]
    for i, review in enumerate(my_reviews_fit.str.slice(0, 100)):
        print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')
        result.append(f'{my_reviews_pred_prob[i]:.2f}:  {review}')
    result_df=pd.Series(result)
    result_df.to_csv('./test/results/result_bert.csv',index=False,header=False)

test_main()