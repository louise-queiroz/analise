#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from os import path, makedirs, chdir, system, getcwd
import json
import glob
print('Diretório:', getcwd())


# ### Parameters

# In[2]:


seed = 94196879

tree_tagger_command = '~/tree-tagger/cmd/tree-tagger-portuguese'

workpath = './'
path_dataset = path.join(workpath, 'chico_processado_data_aspect_hotelManualABSAPT_ternario.csv')
path_test_dataset = path.join(workpath,'chico_processado_sem_correcao_hotelManualABSAPT_ternario.csv')

out_path = path.join(workpath, 'lexicon', 'data')


# In[3]:


print('Diretório:', getcwd())


# ### Create output folder

# In[4]:


makedirs(out_path, exist_ok=True)


# In[5]:


print('Diretório:', getcwd())


# In[6]:


def save_aspects_json(dataset, out_path, variacao_entrada):
    out_file = open(path.join(out_path,f"chico_aspectos{variacao_entrada}.json"), "w") 
    json.dump(dataset.aspect.unique().tolist(), out_file) 
    out_file.close()


# In[7]:


print('Diretório:', getcwd())


# In[8]:


def insert_dumb_columns(dataframe, outfile):
    dataset = dataframe.copy()
    dataset['sub']=''
    dataset['subsub']=''
    dataset = dataset[['review', 'aspect', 'sub', 'subsub', 'polaridade']]
    dataset.to_csv(outfile, sep = ';', index = False, header = None)

# insert_dumb_columns(dataset, out_train_all_features_file)
# insert_dumb_columns(test_dataset, out_test_all_features_file)


# In[9]:


print('Diretório:', getcwd())


# In[10]:


def process_dataset_lexicon(dataset, out_text_only_file):
    dataset_copy = dataset.copy()
    dataset_copy = dataset_copy.drop_duplicates(subset=['review']).sort_values(by=['review']).reset_index(drop=True)
    dataset_copy.index = dataset_copy.index.astype(str)
    dataset_copy.index = 'id_'+dataset_copy.index
    dataset_copy = dataset_copy.drop(columns=['aspect', 'polaridade'])
    dataset_copy.to_csv(out_text_only_file, sep = ';', header = None)
    # print("sed -i 's/s;/ ;/' " + out_text_only_file)
    # system("sed -i 's/;/ ;/ ' " + out_text_only_file.replace(' ', '\ '))#ubuntu
    # system("sed -i 's/;/; /g' " + out_text_only_file.replace(' ', '\ '))#ubuntu
    system("sed -i '' 's/;/ ;/g' '" + out_text_only_file.replace(' ', '\ ') + "'")#mac
    system("sed -i '' 's/;/; /g' '" + out_text_only_file.replace(' ', '\ ') + "'")#mac


# In[11]:


print('Diretório:', getcwd())


# ### Changing output format
# 

# In[12]:


paths_for_inputs = glob.glob(path.join(workpath, f'chico_processado_data_reviews_*.csv'))
dir_original = getcwd()
for path_dataset in paths_for_inputs:
    chdir(dir_original)
    print(f"Processing {path_dataset}")
    variacao_entrada = path_dataset.split('_')[4]
    print(f"Saving aspects in {variacao_entrada}")
    print('Diretório:', getcwd())
    dataset = pd.read_csv(path_dataset, delimiter = ';')
    save_aspects_json(dataset, out_path, variacao_entrada)
    out_train_all_features_file = path.join(out_path, f'chico_AllFeaturesExplicitas_{variacao_entrada}.csv')
    insert_dumb_columns(dataset, out_train_all_features_file)
    out_train_text_only_file = path.join(out_path, f'chico_hotelAll_{variacao_entrada}.txt')
    process_dataset_lexicon(dataset=dataset, out_text_only_file = out_train_text_only_file)
    dir = path.dirname(out_train_text_only_file)
    chdir(dir)
    out_train_tagged = path.join(out_path, f'chico_tagged_hotel_{variacao_entrada}.txt')
    tree_tagger_command_line = f'{tree_tagger_command} {path.basename(out_train_text_only_file)} > {path.basename(out_train_tagged)}'
    print(f'Executing: {tree_tagger_command_line}')
    system(tree_tagger_command_line)
    

