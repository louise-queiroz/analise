#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
pd.options.display.max_columns = None
pd.options.display.max_rows = None


# ### Parameters

# In[2]:


using_neutral = True
neutral_str = ['binario','ternario']

path_datasets = './'
dataset_paths = glob.glob(path_datasets + 'data_reviews_*.csv')


# ### Load corpus

# In[3]:


def load_corpus(path_input):
    dataset = pd.read_csv(path_input, delimiter = ',')
    dataset = dataset.fillna('')
    dataset.drop(columns=['id', 'hash'], inplace=True)
    return dataset


# In[4]:


for path_input in dataset_paths:
    dataset = load_corpus(path_input)
    print("Loading: ", path_input)
    path_output = f'chico_processado_{path_input.split(".")[1].replace("/","")}_ManualABSAPT_{neutral_str[using_neutral]}.csv'
    print("Output path:",path_output)
    dataset.to_csv(path_output, sep = ';', index = None)    


# ### Process Neutral Reviews

# In[5]:


dataset.aspect.sort_values().unique()

