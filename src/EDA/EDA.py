import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import plotly.express as px
import sys
import os
import numpy as np

def eda_report(data):
    '''Te EDA report will create some files to analyze the in deep the variables of the table.
    The elements will be divided by categoric and numeric and some extra info will printed'''
    
    describe_result=data.describe()
    
    eda_path = './files/modeling_output/figures/'

    if not os.path.exists(eda_path):
        os.makedirs(eda_path)

    # Exporting the file
    with open('./files/modeling_output/reports/describe.txt', 'w') as f:
        f.write(describe_result.to_string())

    # Exporting general info
    with open('./files/modeling_output/reports/info.txt','w') as f:
        sys.stdout = f
        data.info()
        sys.stdout = sys.__stdout__
    
    # Rate per genre
    
    fig , ax  = plt.subplots()
    data_rating=data.groupby('genres',as_index=False)['average_rating'].mean().sort_values(by='average_rating',ascending=False).head(10)
    ax.bar(x=data_rating['genres'],height=data_rating['average_rating'])
    ax.set_title('Rating per genre')
    ax.set_xticklabels(data_rating['genres'],rotation=90)
    fig.show()
    fig.savefig(eda_path+'fig.png')
    
    # Votes per genre
    
    fig1 , ax1  = plt.subplots()
    data_voting=data.groupby('genres',as_index=False)['votes'].mean().sort_values(by='votes',ascending=False).head(10)
    ax1.bar(x=data_voting['genres'],height=data_voting['votes'])
    ax1.set_title('Votes per genre')
    ax1.set_xticklabels(data_voting['genres'],rotation=90)
    fig1.show()
    fig1.savefig(eda_path+'fig1.png')
    
    # Timeline
    
    figs, axs = plt.subplots(2, 1, figsize=(16, 8))

    ax = axs[0]

    dft1 = data[['tconst', 'start_year']].drop_duplicates() \
        ['start_year'].value_counts().sort_index()
    dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
    dft1.plot(kind='bar', ax=ax)
    ax.set_title('Número de películas a lo largo de los años')

    ax = axs[1]

    dft2 = data.groupby(['start_year', 'pos'])['pos'].count().unstack()
    dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)

    dft2.plot(kind='bar', stacked=True, label='#reviews (neg, pos)', ax=ax)

    dft2 = data['start_year'].value_counts().sort_index()
    dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
    dft3 = (dft2/dft1).fillna(0)
    axt = ax.twinx()
    dft3.reset_index(drop=True).rolling(5).mean().plot(color='orange', label='reviews per movie (avg over 5 years)', ax=axt)

    lines, labels = axt.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left')

    ax.set_title('Número de reseñas a lo largo de los años')

    figs.tight_layout()
    figs.savefig(eda_path+'figs.png')
    
    # Reviews per movie 
    
    fig3, ax3 = plt.subplots(1, 2, figsize=(16, 5))

    ax = ax3[0]
    dft = data.groupby('tconst')['review'].count() \
        .value_counts() \
        .sort_index()
    dft.plot.bar(ax=ax)
    ax.set_title('Gráfico de barras de #Reseñas por película')

    ax = ax3[1]
    dft = data.groupby('tconst')['review'].count()
    sns.kdeplot(dft, ax=ax)
    ax.set_title('Gráfico KDE de #Reseñas por película')

    fig3.tight_layout()
    fig3.savefig(eda_path+'fig3.png')
    
    # Distribuciones de puntuacion
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    ax = axs[0]
    dft =data.query('ds_part == "train"')['rating'].value_counts().sort_index()
    dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
    dft.plot.bar(ax=ax)
    ax.set_ylim([0, 5000])
    ax.set_title('El conjunto de entrenamiento: distribución de puntuaciones')

    ax = axs[1]
    dft = data.query('ds_part == "test"')['rating'].value_counts().sort_index()
    dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
    dft.plot.bar(ax=ax)
    ax.set_ylim([0, 5000])
    ax.set_title('El conjunto de prueba: distribución de puntuaciones')

    fig.tight_layout()
    fig.savefig(eda_path+'fig4.png')
    
    # Reviews during the year
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=(2, 1), height_ratios=(1, 1)))

    ax = axs[0][0]

    dft = data.query('ds_part == "train"').groupby(['start_year', 'pos'])['pos'].count().unstack()
    dft.index = dft.index.astype('int')
    dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
    dft.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('El conjunto de entrenamiento: número de reseñas de diferentes polaridades por año')

    ax = axs[0][1]

    dft = data.query('ds_part == "train"').groupby(['tconst', 'pos'])['pos'].count().unstack()
    sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
    sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
    ax.legend()
    ax.set_title('El conjunto de entrenamiento: distribución de diferentes polaridades por película')

    ax = axs[1][0]

    dft = data.query('ds_part == "test"').groupby(['start_year', 'pos'])['pos'].count().unstack()
    dft.index = dft.index.astype('int')
    dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
    dft.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('El conjunto de prueba: número de reseñas de diferentes polaridades por año')

    ax = axs[1][1]

    dft = data.query('ds_part == "test"').groupby(['tconst', 'pos'])['pos'].count().unstack()
    sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
    sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
    ax.legend()
    ax.set_title('El conjunto de prueba: distribución de diferentes polaridades por película')

    fig.tight_layout()
    fig.savefig(eda_path+'fig5.png')