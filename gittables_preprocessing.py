import pickle
import os
import pandas as pd
import tqdm
import sys

def list_files(directory) -> list:
    """given the path of a directory return the list of its files

    Args:
        directory (_type_): path to the directory to explore

    Returns:
        list: list of filenames
    """
    l=[]
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                l.append(entry.name)
    return l

def generate_table_dict(gittables_folder_path: str, outpath: str=None, log_path: str=None) -> dict:
    """_summary_

    Args:
        gittables_folder_path (str): path to the folder containing the tables in the csv format
        outpath (str): folder where to save the generated table_dict
        log_path (str, optional): path to the directory where to save the names of the dropped tables. Defaults to None.

    Returns:
        dict: a table_dict
    """
    filenames = list_files(gittables_folder_path)
    log = []
    table_dict = {}
    for i in tqdm.tqdm(range(len(filenames))):
        t = None
        path = gittables_folder_path + '/' + filenames[i]
        try:
            t = pd.read_csv(path, sep=',', header=None)
        except:
            try:
                t = pd.read_csv(path, sep='#', header=None)
            except:
                log.append(path)
        if isinstance(t, pd.DataFrame):
            table_dict[str(filenames[i])] = t
        

    if log_path:
        with open(log_path, 'w') as file:
            # Write the string to the file
            file.write('\n'.join(log))
    if isinstance(outpath, str):
        with open(outpath, 'wb') as f:
            pickle.dump(table_dict, f)
    return table_dict

def generate_table_dict_from_different_folders(folders: list, outpath: str) -> dict:
    table_dict = {}
    for d in folders:
        table_dict.update(generate_table_dict(d))
    with open(outpath, 'wb') as f:
        pickle.dump(table_dict, f)

if __name__ == '__main__':
    
    # n_params = len(sys.argv) - 1
    # expected_params = 2
    # if (n_params != expected_params) and (n_params != (expected_params+1)):
    #     raise ValueError(f'Wrong number of parameters, you provided {n_params} but {expected_params} or {expected_params+1} are expected. \nUsage is: {sys.argv[0]} gittables_folder_path out_file_path [log_path]')
    # gittables_path = sys.argv[1]
    # out_file_path = sys.argv[2]
    # log_path = sys.argv[3]
    # generate_table_dict(gittables_path,out_file_path,log_path)
    #generate_table_dict("/data/gittables/csv","/home/francesco.pugnaloni/GNNTE/Datasets/gittables_datasets/gittables_full.pkl","/home/francesco.pugnaloni/GNNTE/Datasets/gittables_datasets/logs.txt")

    generate_table_dict_from_different_folders(['/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/csv_tables/train', '/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/csv_tables/test', '/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/csv_tables/valid'], 
                                               '/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/table_dict.pkl')