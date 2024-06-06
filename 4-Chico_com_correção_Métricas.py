#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import os
from  sklearn.metrics import *


# In[2]:


def get_metrics(target, predict):
    metrics = {}
    metrics['bacc'] = balanced_accuracy_score(target, predict)
    metrics['accuracy'] = accuracy_score(target, predict)
    metrics['f1_macro'] = f1_score(target, predict, average='macro')
    metrics['f1_micro'] = f1_score(target, predict, average='micro')
    metrics['f1_weighted'] = f1_score(target, predict, average='weighted')
    metrics['precision_macro'] = precision_score(target, predict,average='macro')
    metrics['precision_micro'] = precision_score(target, predict,average='micro')
    metrics['precision_weighted'] = precision_score(target, predict,average='weighted')
    metrics['recall_macro'] = recall_score(target, predict,average='macro')
    metrics['recall_micro'] = recall_score(target, predict,average='micro')
    metrics['recall_weighted'] = recall_score(target, predict,average='weighted')
    return metrics,f1_score(target, predict, average=None)


# In[3]:


workpath = './'
experiments_path = 'output'

paths_for_inputs = glob.glob(os.path.join(workpath, f'chico_processado_data_reviews_*.csv'))
corretores = []
for path_dataset in paths_for_inputs:
    variacao_entrada = path_dataset.split('_')[4]
    corretores.append(variacao_entrada)

print(f'Corretores: {corretores}')

lexicos = ['AffectPT_br_editado', 'AffectPT_br', 'EmoLex', 'LeIA', 'LIWC', 'OntoPT', 'OpLexicon', 'ReLi_Lex', 'SentiLex', 'SentiWordNet', 'UNILEX', 'Wordnet_Affect_BR']

for corretor in corretores:
    arquivoComMarcacaoManual = f'lexicon/data/chico_AllFeaturesExplicitas_{corretor}.csv'

    manual = pd.read_csv(arquivoComMarcacaoManual, sep=';', header=None)
    
    experiments = glob.glob(f'{experiments_path}/{corretor}/*/fold_0/chico_predicoes_freitas_treetagger_{corretor}_*.csv')

    for arquivoComMarcacaoAutomatica in experiments:
        freitas = pd.read_csv(arquivoComMarcacaoAutomatica, sep=';', header=None)
        experiments_com_neutro = glob.glob(f'{experiments_path}/{corretor}/*/fold_0/Freitas_com_neutro.csv')
        metrics_com_neutro = {}
        for completo_com_neutro in experiments_com_neutro:
            lexicon_name = completo_com_neutro.split('/')[-3] 
            df_completo_com_neutro = pd.read_csv(completo_com_neutro, sep=';')
            metrics_com_neutro[lexicon_name] = get_metrics(df_completo_com_neutro['target'], df_completo_com_neutro['predict'])[0]
        
        pd.DataFrame(metrics_com_neutro).T.sort_values('bacc', ascending=False).to_csv(f'output/metrics_com_neutro_{corretor}_{lexicon_name}.csv')

        experiments_sem_neutro = glob.glob(f'{experiments_path}/{corretor}/*/fold_0/Freitas_sem_neutro.csv')
        metrics_sem_neutro = {}
        for completo_sem_neutro in experiments_sem_neutro:
            lexicon_name = completo_sem_neutro.split('/')[-3] 
            df_completo_sem_neutro = pd.read_csv(completo_sem_neutro, sep=';')
            metrics_sem_neutro[lexicon_name] = get_metrics(df_completo_sem_neutro['target'], df_completo_sem_neutro['predict'])[0]
            
        pd.DataFrame(metrics_sem_neutro).T.sort_values('bacc', ascending=False).to_csv(f'output/metrics_sem_neutro_{corretor}_{lexicon_name}.csv')


# In[4]:


pd.DataFrame(metrics_com_neutro).T.sort_values('bacc', ascending=False)


# In[5]:


pd.DataFrame(metrics_sem_neutro).T.sort_values('bacc', ascending=False)


# In[ ]:




