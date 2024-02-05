import tqdm
import pandas as pd
from GNNTE import set_seed
from my_constants import *
from random import randint
from sklearn.model_selection import train_test_split
import pickle

def group_samples(samples_path: str | pd.DataFrame) -> dict:
    if type(samples_path) == str:
        samples = pd.read_csv(samples_path)
    else:
        samples = samples_path

    d = {}
    for i in tqdm.tqdm(range(samples.shape[0])):
        t1 = samples.iloc[i][0]
        t2 = samples.iloc[i][1]
        try:
            d[t1].append(i)
        except:
            d[t1] = []
            d[t1].append(i)
        try:
            d[t2].append(i)
        except:
            d[t2] = []
            d[t2].append(i)
    for k in d.keys():
        d[k] = set(d[k])
    return d

def train_test_valid_split(table_indexes, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, seed = 42) -> set:
    train, test = train_test_split(table_indexes, test_size=test_ratio, random_state=seed)
    train, valid = train_test_split(train, test_size=validation_ratio / (1 - test_ratio), random_state=seed)
    return set(train), set(test), set(valid)

def generate_splits_from_tables(sample_file: str, outdir: str=None, tables_to_samples: str=None) -> None:
    samples = pd.read_csv(sample_file)
    if tables_to_samples:
        with open(tables_to_samples, 'rb') as f:
            tables_to_samples = pickle.load(f)
    else:
        tables_to_samples = group_samples(samples)

    table_names = list(tables_to_samples.keys())
    train, test, valid = train_test_valid_split(table_names)

    train_samples = []
    test_samples = []
    valid_samples = []

    for i in tqdm.tqdm(range(samples.shape[0])):
        if (samples.iloc[i][0] in train) and (samples.iloc[i][1] in train):
            train_samples.append(i)
        elif (samples.iloc[i][0] in test) and (samples.iloc[i][1] in test):
            test_samples.append(i)
        elif (samples.iloc[i][0] in valid) and (samples.iloc[i][1] in valid):
            valid_samples.append(i)
    
    tot_samples = len(train_samples)+len(test_samples)+len(valid_samples)

    train_set = samples.iloc[train_samples][:]
    test_set = samples.iloc[test_samples][:]
    valid_set = samples.iloc[valid_samples][:]

    if outdir:
        train_set.to_csv(outdir+'/train.csv',index=False)
        test_set.to_csv(outdir+'/test.csv',index=False)
        valid_set.to_csv(outdir+'/valid.csv',index=False)

    print(f'Maintained  {tot_samples}/{samples.shape[0]} samples\nTrain: {len(train_samples)}\nTest: {len(test_samples)}\nValid: {len(valid_samples)}')
    

if __name__ == '__main__':
    generate_splits_from_tables('/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/1M_wikitables_disjointed/all_samples.csv', 
                                outdir='/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/1M_wikitables_disjointed',
                                tables_to_samples='/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/1M_wikitables_disjointed/tables_to_sample.pkl')
    print('finish')