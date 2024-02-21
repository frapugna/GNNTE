import pickle
import pandas as pd
from graph import is_float
from tqdm import tqdm
from graph import *

def process_table(table: pd.DataFrame, sp: String_token_preprocessor) -> dict:
    """Method to process a single table and compute some stats

    Args:
        table (pd.DataFrame): table to process

    Returns:
        dict: updated version of the dictionary containing stats about the processed table
    """
    n_rows = table.shape[0]
    n_cols = table.shape[1]
    area = n_rows * n_cols

    n_num = 0
    n_text = 0
    n_nan = 0
    
    is_text = False
    is_num = False
    has_nan = False
    n_tokens = 0


    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            t = table.iloc[r][c]
            if pd.isnull(t):
                n_nan += 1
                n_tokens += 1
            elif isinstance(t, str) and not(is_float(t)):
                n_text += 1
                tokens = sp(t, operations=['lowercase', 'split'])
                n_tokens += len(tokens)
            elif is_float(str(t)):
                n_num += 1
                n_tokens += 1
            else:
                continue

    if n_num == 0 and n_text != 0:
        is_text = True

    if n_num != 0 and n_text == 0:
        is_num = True

    if n_nan != 0:
        has_nan = True

    percentage_num = n_num / area
    percentage_text = n_text / area
    percentage_nan = n_nan / area

    out = {
        'n_rows':n_rows,
        'n_cols':n_cols,
        'area':area,
        'n_num':n_num,
        'n_text':n_text,
        'n_nan':n_nan,
        'precentage_num':percentage_num,
        'percentage_text':percentage_text,
        'percentage_nan':percentage_nan,
        'is_text':is_text,
        'is_num':is_num,
        'has_nan':has_nan,
        'n_tokens':n_tokens
    }

    return out
    

def compute_tables_stats(table_dict_path: str, out_path: str) -> dict:
    """Given a table_dict compute some relevant stats about the collection

    Args:
        table_dict_path (str): path to a dictionary saved in pickle format that contains the dataframes to analyze
        out_path (str): directory where to save the dictionary containing the stats in pickle format

    Returns:
        dict: the dictionary containing the stats
    """
    sp = String_token_preprocessor()
    with open(table_dict_path, 'rb') as f:
        table_dict = pickle.load(f)

    stats_dict = {}
    for k in tqdm(table_dict.keys()):
        tmp = process_table(table_dict[k], sp)
        if tmp['n_tokens'] != 0:
            stats_dict[k] = tmp 
    

    with open(out_path, 'wb') as f:
        pickle.dump(stats_dict, f)

if __name__ == '__main__':
    #d = compute_tables_stats("/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_tables.pkl",'/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_tables_stats.pkl')
    #d = compute_tables_stats("/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_tables.pkl",'/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_tables_stats.pkl')
    #d = compute_tables_stats("/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1MR/full_table_dict_with_id.pkl",'/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1MR/stats.pkl')
    d = compute_tables_stats("/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/full_table_dict_with_id.pkl",'/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/stats/stats.pkl')