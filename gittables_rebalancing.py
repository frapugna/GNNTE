import pandas as pd
from data_visualization import *
from tqdm import tqdm

def drop_from_interval(df: pd.DataFrame, min_interval: float, max_interval: float, n_samples_to_drop: int) -> pd.DataFrame:
    tmp = df[df['a%'] >= min_interval]
    if max_interval == 1:
        tmp = tmp[tmp['a%'] <= max_interval]
    else:
        tmp = tmp[tmp['a%'] < max_interval]
    to_drop_rows = tmp.sample(n_samples_to_drop)
    to_drop = list(to_drop_rows.index)

    return df.drop(to_drop), to_drop_rows

def augment_interval(df_old: pd.DataFrame, df_for_augmentation: pd.DataFrame, min_interval: float, max_interval: float, n_samples_to_add: int) -> pd.DataFrame:
    tmp = df_for_augmentation[df_for_augmentation['a%'] >= min_interval]
    if max_interval == 1:
        tmp = tmp[tmp['a%'] <= max_interval]
    else:
        tmp = tmp[tmp['a%'] < max_interval]
    tmp = tmp.sample(n_samples_to_add)
    result = pd.concat([df_old, tmp], axis=0)
    return result

def get_table_set(df: pd.DataFrame) -> set:
    out = []
    for r in tqdm(range(df.shape[0])):
        out.append(df.iloc[r]['r_id'])
        out.append(df.iloc[r]['s_id'])
    return set(out)

def drop_repetitions(t_not_augmented: pd.DataFrame, augmentation_data: pd.DataFrame) -> pd.DataFrame:
    table_set_not_augmented = get_table_set(t_not_augmented)
    table_set_augmentation_data = get_table_set(augmentation_data)
    tables_to_drop = []
    for t in tqdm(table_set_augmentation_data):
        if t in table_set_not_augmented:
            tables_to_drop.append(t)
    
    tables_to_drop = set(tables_to_drop)
    
    rows_to_drop = []

    for r in tqdm(range(augmentation_data.shape[0])):
        if (augmentation_data.iloc[r]['r_id'] in tables_to_drop) or (augmentation_data.iloc[r]['s_id'] in tables_to_drop):
            rows_to_drop.append(r)

    return augmentation_data.drop(rows_to_drop)

def check_no_repetitions(s1: set, s2: set) -> bool:
    for t in tqdm(s1):
        if t in s2:
            return True
    return False

def extract_suitable_triples(triples_ok: pd.DataFrame | str, triples_to_filter_list: pd.DataFrame | str, outpath: str=None) -> pd.DataFrame:
    if isinstance(triples_ok, str):
        triples_ok = pd.read_csv(triples_ok)
    set_ok = get_table_set(triples_ok)
    out = None
    for triples_to_filter in triples_to_filter_list:
        print(f'Exploring: {triples_to_filter}')
        if isinstance(triples_to_filter, str):
            triples_to_filter = pd.read_csv(triples_to_filter)
        ok_rows_list = []

        for r in tqdm(range(triples_to_filter.shape[0])):
            if (triples_to_filter.iloc[r]['r_id'] in set_ok) and (triples_to_filter.iloc[r]['s_id'] in set_ok):
                ok_rows_list.append(r)
        
        
        if isinstance(out, pd.DataFrame):
            out = pd.concat([out, triples_to_filter.iloc[ok_rows_list]], axis=0)
        else:
            out = triples_to_filter.iloc[ok_rows_list]

    if outpath:
        out.to_csv(outpath, index=False)
    
    return out

if __name__ == '__main__':
    extract_suitable_triples(triples_ok='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/balanced_datasets/unbalanced_datasets/valid_unbalanced.csv',
                             triples_to_filter_list=['/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/labelled/old_data/train_stats_cleaned.csv',
                                                     '/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/labelled/old_data/test_stats_cleaned.csv',
                                                     '/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/labelled/old_data/valid_stats_cleaned.csv'],
                             outpath='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/balanced_datasets/data_augementation_samples/valid_data_augm.csv'
                             )