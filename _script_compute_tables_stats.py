import pickle
import pandas as pd
from graph import is_float
import tqdm

def process_table(table: pd.DataFrame, state_dict: dict) -> dict:
    """Method to process a single table and compute some stats

    Args:
        table (pd.DataFrame): table to process
        state_dict (dict): dictionary containing the stats computed so far

    Returns:
        dict: updated version of the dictionary containing stats about the processed table
    """
    state_dict['rows_per_table'].append(table.shape[0])
    state_dict['columns_per_table'].append(table.shape[1])
    state_dict['area_per_table'].append(table.shape[0] * table.shape[1])

    n_text = 0
    lens_text = []
    n_num = 0
    n_nan = 0

    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            t = table.iloc[r][c]
            if pd.isnull(t):
                n_nan += 1
            elif isinstance(t, str) and not(is_float(t)):
                n_text += 1
                lens_text.append(len(t))
            elif is_float(str(t)):
                n_num += 1
            else:
                continue
    
    state_dict['textual_cells_per_table'].append(n_text)
    state_dict['numerical_cells_per_table'].append(n_num)
    if len(lens_text) == 0:
        state_dict['avg_length_textual_cells_per_table'].append(0)
    else:
        state_dict['avg_length_textual_cells_per_table'].append(sum(lens_text)/len(lens_text))
    state_dict['textual_cells_per_table'].append(n_nan)

    return state_dict
    

def compute_tables_stats(table_dict_path: str, out_path: str) -> dict:
    """Given a table_dict compute some relevant stats about the collection

    Args:
        table_dict_path (str): path to a dictionary saved in pickle format that contains the dataframes to analyze
        out_path (str): directory where to save the dictionary containing the stats in pickle format

    Returns:
        dict: the dictionary containing the stats
    """
    with open(table_dict_path, 'rb') as f:
        table_dict = pickle.load(f)
    index_to_key = list(table_dict.keys())
    stats_dict = {
        'rows_per_table':[],
        'columns_per_table':[],
        'area_per_table':[],
        'textual_cells_per_table':[],
        'numerical_cells_per_table':[],
        'avg_length_textual_cells_per_table':[],
        'nan_per_table':[]
    }
    for i in tqdm.tqdm(range(len(index_to_key))):
    #for i in tqdm.tqdm(range(100)):
        stats_dict = process_table(table_dict[str(index_to_key[i])], stats_dict)

    n_tables = len(index_to_key)

    stats_dict['number_of_tables'] = n_tables
    
    stats_dict['max_rows'] = max(stats_dict['rows_per_table'])
    stats_dict['min_rows'] = min(stats_dict['rows_per_table'])
    stats_dict['avg_rows_per_table'] = sum(stats_dict['rows_per_table']) / n_tables

    stats_dict['max_columns'] = max(stats_dict['columns_per_table'])
    stats_dict['min_columns'] = min(stats_dict['columns_per_table'])
    stats_dict['avg_columns_per_table'] = sum(stats_dict['columns_per_table']) / n_tables

    stats_dict['max_area'] = max(stats_dict['area_per_table'])
    stats_dict['min_area'] = min(stats_dict['area_per_table'])
    stats_dict['avg_area'] = sum(stats_dict['area_per_table']) / n_tables

    stats_dict['avg_textual_cells_per_table'] = sum(stats_dict['textual_cells_per_table']) / n_tables
    stats_dict['avg_numerical_cells_per_table'] = sum(stats_dict['numerical_cells_per_table']) / n_tables
    stats_dict['avg_length_textual_cells'] = sum(stats_dict['avg_length_textual_cells_per_table']) / n_tables
    stats_dict['avg_nan_per_table'] = sum(stats_dict['nan_per_table']) / n_tables

    stats_dict['index_to_key'] = index_to_key

    with open(out_path, 'wb') as f:
        pickle.dump(stats_dict, f)

if __name__ == '__main__':
    #d = compute_tables_stats("/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_tables.pkl",'/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_tables_stats.pkl')
    #d = compute_tables_stats("/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_tables.pkl",'/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_tables_stats.pkl')
    #d = compute_tables_stats("/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1MR/full_table_dict_with_id.pkl",'/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1MR/stats.pkl')
    d = compute_tables_stats("/home/francesco.pugnaloni/GNNTE/Datasets/gittables_datasets/gittables_full.pkl",'/home/francesco.pugnaloni/GNNTE/Datasets/gittables_datasets/gittables_stats.pkl')