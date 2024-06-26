from typing import Optional
import pandas as pd
from tqdm import tqdm
from statistics import mean
import random

def deduplicate_hybrid(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate the dataset containing non exact table overlaps

    Args:
        df (pd.DataFrame): path to the old triple dataset

    Returns:
        pd.DataFrame: the new triple dataset
    """
    df = df.sort_values('r_id')
    to_drop = []
    i = 0
    while i < df.shape[0]-1:
        if (df['r_id'][i] == df['r_id'][i+1]) and df['s_id'][i] == df['r_id'][i+1]:
            if df['algo'][i] == 'a':
                to_drop.append(i)
            if df['algo'][i+1] == 'a':
                to_drop.append(i+1)
                i+=1
        i+=1
    return df.drop(to_drop)    

def prepare_triple_file(input_file: str, output_file: str, is_hybrid=False) -> None:
    """
        ____Deprecated____
    """
    try:
        df = pd.read_csv(input_file)
    except:
        raise Exception('Wrong input file path')
    print('Input file loaded') 
    df['table_overlap'] = df['o_a'] / df[['r_a','s_a']].min(axis=1)
    if is_hybrid:
        df_out = df[['r_id','s_id','table_overlap','algo']]
        df_out = deduplicate_hybrid(df_out)
    else:
        df_out = df[['r_id','s_id','table_overlap']]
    print('Output file generated')
    try:
        df_out.to_csv(output_file, index=False)
    except:
        raise Exception('Write operation failed') 
    print('Write operation succeded')

def rebalance_triple_file(input_file: str, output_file: str,thresholds: Optional[int]=None, set_thresholds: bool=False, drop_1: bool=True) -> pd.DataFrame:
    """Balance the overlap distribution in the dataset

    Args:
        input_file (str): path to the unbalanced dataset
        output_file (str): path to the file where to save the balanced dataset
        thresholds (Optional[int], optional): __Not implemented__. Defaults to None.
        set_thresholds (bool, optional): __Not implemented. Defaults to False.
        drop_1 (bool, optional): drop all the triple with table overlap equal to 1. Defaults to True.

    Raises:
        Exception: raised if the path to the dataset file is wrong
        Exception: NotImplemented
        NotImplementedError: NotImplemented
        Exception: raised if the write operation fails

    Returns:
        pd.DataFrame: the rebalanced dataset
    """
    try:
        df = pd.read_csv(input_file)
    except:
        raise Exception('Wrong input file path')
    
    if set_thresholds:
        if len(thresholds)!=10:
            raise Exception('The thresholds set must contain exactly 10 values')
        occs = []
        for i in range(len(thresholds)):
            occs.append([])
        raise NotImplementedError
    
    if drop_1:
        df_out = df[df['table_overlap']!=1]
    show_samples_distribution(df_out)
    try:
        df_out.to_csv(output_file, index=False)
    except:
        raise Exception('Write operation failed') 
    print('Write operation succeded')
    return df_out

def generate_thresholded_dataset(path_in:str,  path_out:str, granularity:float=0.1, strategy:str='min') -> None:
    """The dataset is divided in bins and a threshold is applied on their numbers of elements

    Args:
        path_in (str): path to the input dataset
        path_out (str): path to the file where to save the new dataset
        granularity (float, optional): size of the bins (percentual). Defaults to 0.1.
        strategy (str, optional): thresholding strategy {'min', 'mean'}. Defaults to 'min'.

    Raises:
        Exception: raised if the write operation fails
    """
    df = pd.read_csv(path_in)
    d = {}
    print('Bins computation starts')
    for i in tqdm(df['table_overlap']):
        n = i//granularity/10
        if i == 1:
            n = 1
        try:
            d[n]+=1
        except:
            d[n]=1
    if strategy=='min':
        thershold = min(d.values())
    if strategy=='mean':
        thershold = mean(d.values())
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    occ_counts = {}
    index_out = []
    print('Thresholded dataset computation starts')
    for i in tqdm(range(df.shape[0])):
        n = df['table_overlap'][i] //granularity/10
        try:
            occ_counts[n] += 1
        except:
            occ_counts[n] = 0
        if occ_counts[n] < thershold:
            index_out.append(i)
    df=df.iloc[index_out][:]
    try:
        df.to_csv(path_out, index=False)
    except:
        raise Exception('Write operation failed') 
    print('Write operation succeded')

def generate_full_triple_dataset(path_base: str, path_hybrid: str, path_out: str) -> pd.DataFrame:
    """_summary_

    Args:
        path_base (str): path to the clean triple dataset
        path_hybrid (str): path to the dirty triple dataset
        path_out (str): path to the file where to save the new triple dataset

    Raises:
        Exception: raised if the clen path is wrong
        Exception: raised if the dirty path is wrong
        Exception: raised if the write operation fails

    Returns:
        pd.DataFrame: the new triple dataset
    """
    try:
        df_base = pd.read_csv(path_base)
    except:
        raise Exception('Wrong input file path')
    print('Input file base loaded') 
    df_base['table_overlap'] = df_base['o_a'] / df_base[['r_a','s_a']].min(axis=1)
    df_base = df_base[['r_id','s_id','table_overlap']]
    try:
        df_hybrid = pd.read_csv(path_hybrid)
    except:
        raise Exception('Wrong input file path')   
    
    print('Input file base processed')
    df_hybrid['table_overlap'] = df_hybrid['o_a'] / df_hybrid[['r_a','s_a']].min(axis=1)
    df_hybrid = df_hybrid[['r_id','s_id','table_overlap','algo']]
    print('Input file hybrid loaded, deduplication is starting...')
    df_hybrid = deduplicate_hybrid(df_hybrid)[['r_id','s_id','table_overlap']]
    print('Deduplication succeded, input file hybrid processed')
    df_out = pd.concat([df_base, df_hybrid])
    
    try:
        df_out.to_csv(path_out, index=False)
    except:
        raise Exception('Write operation failed') 

    print('Output file generated')

    return df_out

def generate_csv_min_mean(path_in: str, out_directory: str, agg: list=['min','mean'], gran: list=[0.1,0.01], name: list=['01','001']) -> None:
    """Generate datasets using different strategies

    Args:
        path_in (str): path to the old dataset
        out_directory (str): path to the directory where to save the new datasets
        agg (list, optional): list of strategies to use. Defaults to ['min','mean'].
        gran (list, optional): list of granularities to use. Defaults to [0.1,0.01].
        name (list, optional): names. Defaults to ['01','001'].
    """
    for i in range(len(gran)):
        for j in range(len(agg)):
            generate_thresholded_dataset(path_in=path_in,
                                path_out=f"{out_directory}/test_samples_thresholded_{name[i]}_{agg[j]}.csv",
                                granularity=gran[i],
                                strategy=agg[j])
            
def extract_exact_overlap(df: pd.DataFrame, limit: int, threshold: float) -> pd.DataFrame:
    """Generate a new dataset

    Args:
        df (pd.DataFrame): the full triple dataset
        limit (int): limit number
        threshold (float): similarity threshold

    Returns:
        pd.DataFrame: the new dataset
    """
    count = 0
    out_list = []
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    for i in range(df.shape[0]):
        if df['table_overlap'][i]==threshold:
            out_list.append(i)
            count+=1
            if count >= threshold:
                break
    return df.iloc[out_list][:]

def show_samples_distribution(df:pd.DataFrame, granularity:float=0.1)->dict:
    """The dataset is divided in bins based on sample's table overlap, a bar diagram is displayed to show visually the data distribution

    Args:
        df (pd.DataFrame): the dataframe to analyze
        granularity (float, optional): the size of the bins. Defaults to 0.1.

    Returns:
        dict: contains the count of elements in every bin
    """
    d = {}
    for i in tqdm(df['table_overlap']):
        n = i//granularity/10
        if i == 1:
            n = 1
        try:
            d[n]+=1
        except:
            d[n]=1
    l=[ [k,v] for k,v in d.items()]
    df_occurrencies = pd.DataFrame(l).sort_values(0)
    ax = df_occurrencies.plot(x=0, y=1, kind='bar')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')
    return d

def describe_samples_distribution(df: pd.DataFrame, granularity: float=0.1) -> dict:
    """The dataset is divided in bins based on sample's table overlap

    Args:
        df (pd.DataFrame): the dataframe to analyze
        granularity (float, optional): the sizeo of the bins. Defaults to 0.1.

    Returns:
        dict: a dictionary which contains the "describes" of all the bins
    """
    d = {}
    for i in tqdm(range(len(df))):
        n = df['table_overlap'][i]//granularity/10
        if i == 1:
            n = 1
        try:
            d[n].append(i)
        except:
            d[n]=[]
            d[n].append(i)
    out = {}
    for k in d.keys():
        print(f'Bin: {k}')
        print(df['table_overlap'][d[k]].describe())
        out[k]=df['table_overlap'][d[k]]
    
    return out

def re_generate_triple_datasets(input_file_base: str, input_file_hybrid: str, out_directory: str) -> None:
    
    """This function perform all the preprocessing pipeline necessary to obtain the triples file necesessary for the training of the model:
    * test_samples_dirty.csv
    * test_samples_no_ones.csv
    * test_samples_base.csv 
    * test_samples_thresholded_001_mean.csv
    * test_samples_thresholded_001_min.csv
    * test_samples_thresholded_01_mean.csv
    * test_samples_thresholded_01_min.csv

    Args:
        input_file_base (str): path to the raw file containing all the matches with their overlap
        input_file_hybrid (str): path to the raw file containing all the approximated matches with their overlap
        out_directory (str): directory where to save all the generated csv files
    """
    print('Full triple dataset generation starts')
    df_full = generate_full_triple_dataset(input_file_base, input_file_hybrid, out_directory+"/test_samples_dirty.csv")
    print('Full triple dataset generation ends')
    
    print('Perfect matches extraction operation starts')
    df_perfect_matches = extract_exact_overlap(df_full, 10000, 1)
    print('Perfect matches extraction operation ends')

    print('Rebalancing operation starts')
    df_no_1 = rebalance_triple_file(out_directory+"/test_samples_dirty.csv", out_directory+"/test_samples_no_ones.csv")
    print('Rebalancing operation ends')

    print('Adding perfect matches to the dataset')
    df_full = pd.concat([df_no_1, df_perfect_matches])
    try:
        df_full.to_csv(out_directory+"/test_samples_base.csv", index=False)
    except:
        raise Exception('Write operation failed') 
    
    print('Thresholded datasets generation starts')
    generate_csv_min_mean(out_directory+"/test_samples_base.csv", out_directory)
    print('Thresholded datasets generation ends')

# def generate_tables_to_triples(triple_file: str) -> dict:
#     triples = pd.read_csv(triple_file)


# def generate_train_test_valid_split(tables_to_triples: str, triple_file: str, p_train: float=0.8, p_test: str=0.1, p_valid: float=0.1):
#     pass


if __name__=='__main__':
    re_generate_triple_datasets("/dati/home/francesco.pugnaloni/wikipedia_tables/unprocessed_tables/triples_wikipedia_tables.csv",
                                "/dati/home/francesco.pugnaloni/wikipedia_tables/unprocessed_tables/hybrid_dataset_stats.csv",
                                "/home/francesco.pugnaloni/wikipedia_tables/processed_tables"
                                )
    print('ok')