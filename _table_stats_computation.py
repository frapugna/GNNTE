import pickle
import pandas as pd
from graph import is_float
from tqdm import tqdm
from graph import *
import numbers

def compute_counts(t: pd.DataFrame) -> tuple:
    value_count = {}
    n_nans = 0
    n_str = 0
    n_num = 0
    n_bools = 0

    for r in range(t.shape[0]):
        for c in range(t.shape[1]):
            v = t.iloc[r][c]
            value_count[v] = 0
            if pd.isna(v):
                n_nans += 1
            elif isinstance(v, bool):
                n_bools += 1
            elif isinstance(v, numbers.Number):
                n_num += 1
            else:
                n_str += 1
    return len(value_count.keys()), n_num, n_str, n_nans, n_bools

def compute_tables_stats(table_dict: dict | str, outpath: str) -> None:
    if isinstance(table_dict, str):
        with open(table_dict, 'rb') as f:
            table_dict = pickle.load(f)
    new_cols = {
        'table_name':[],
        'n_cols':[],
        'n_rows':[],
        'area':[],
        'n_distinct_values':[],
        'n_numerical':[],
        'n_textual':[],
        'n_nans':[],
        'n_bools':[]
    }

    for k in tqdm(table_dict.keys()):
        t = table_dict[k]
        rows = t.shape[0]
        cols = t.shape[1]
        area = rows * cols
        n_distinct_values, n_numerical, n_textual, n_nans, n_bools = compute_counts(t)
        new_cols['table_name'].append(k)
        new_cols['n_cols'].append(cols)
        new_cols['n_rows'].append(rows)
        new_cols['area'].append(area)
        new_cols['n_distinct_values'].append(n_distinct_values)
        new_cols['n_numerical'].append(n_numerical)
        new_cols['n_textual'].append(n_textual)
        new_cols['n_nans'].append(n_nans)
        new_cols['n_bools'].append(n_bools)
    
    out = pd.DataFrame(new_cols)
    out.to_csv(outpath, index=False)
    print(out.describe())

