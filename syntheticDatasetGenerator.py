import pandas as pd
import random
from graph import *
from trainingTablesPreprocessing import *
from _script_generate_graph_dict import generate_graph_dictionary
import pickle
import time

def extract_token_list_to_populate_tables(table_dict_path: str, outpath: str, number_of_tokens: int=1000000, token_limit_per_table: int=15) -> None:
    """
    Extract a list of tokens from an input table_dict.
    Args:
        table_dict_path (str): source dictionary containing pandas dataframes.
        outpath (str): file where to save the token_list in .pkl format.
        number_of_tokens (int, optional): desired number of total tokens. Defaults to 50000.
        token_limit_per_table (int, optional): upper bound to how many tokens can be extracted from a single table. Defaults to 10.
    """
    print("Starting")
    with open(table_dict_path, 'rb') as f:
        table_dict = pickle.load(f)
    out = []
    token_count = 0
    for k in table_dict.keys():
        print(f'Found {token_count} tokens')
        t = table_dict[k]
        table_token_count = 0
        for r in range(t.shape[0]):
            for c in range(t.shape[1]):
                value = t.iloc[r][c]
                if pd.isnull(value):
                    continue
                out.append(value)
                token_count+=1
                table_token_count+=1
            if table_token_count >= token_limit_per_table:
                break
        if token_count >= number_of_tokens:
            break
    print('saving')
    with open(outpath, 'wb') as f:
        pickle.dump(out, f)


def generateDatasetList(n_datasets: int, max_n_cols: int, max_n_rows: int, sentences: list=None) -> list:
    """Function that generates a new list of datasets

    Args:
        n_datasets (int): nmber of datasets to generate
        max_n_cols (int): max number of cols of the datasets
        max_n_rows (int): max number of rows of the datasets
        sentences (list, optional): list of possible values to put inside the datasets. Defaults to None.

    Returns:
        list: the list of datasets
    """
    if not(sentences):
        sentences = ['I have a pen and an apple','this sentence will be pretty long and hopefully it will confuse bert with a lot of useless words that are not only stop wrods but also other stuff and yws this sentence is pretty long again',
                  'cat', 'dog','Keanu Reeves is an actor 3', '4','1','2','3','4','5','6','7','8','9','0','I believe I can fly', 'The plot of the book is that it is a book', 'Leopard','JaGuar',
                  'this sentence will be pretty long and hopefully it will confuse bert with a lot of useless words that are not only stop wrods but also other stuff and yws this sentence is pretty long again',
                  'this sentence will be pretty long and hopefully it will confuse bert with a lot of useless words that are not only stop wrods but also other stuff and yws this sentence is pretty long again']
    n_sentences = len(sentences)
    tables = []
    for _ in range(n_datasets):
        r = random.randint(1,max_n_rows)
        c = random.randint(1,max_n_cols)
        rows = []
        for i in range(r):
            row = []
            for j in range(c):
                n = random.randint(0,n_sentences-1)
                row.append(sentences[n])
            rows.append(row)
        tables.append(pd.DataFrame(rows))
    return tables

def processDataset(df: pd.DataFrame, embedding_buffer: Embedding_buffer, string_token_preprocessor: String_token_preprocessor) -> float:
    """Function to check how fast the graph is generated

    Args:
        df (pd.DataFrame): dataset to process
        embedding_buffer (Embedding_buffer): an instance of embedding buffer
        string_token_preprocessor (String_token_preprocessor): an instance of string_token_preprocessor

    Returns:
        float: the execution time
    """
    start = time.time()
    Graph(df, 'Table1', embedding_buffer, string_token_preprocessor, verbose=False,token_length_limit=None)
    end = time.time()
    tot = end-start
    print(f'T_exec: {tot}s')
    return tot

def generate_test_training_stuff(num_tables: int, num_samples: int, 
                                 outdir: str="/home/francesco.pugnaloni/wikipedia_tables/small_dataset_debug",
                                 w: int=10, h: int=10, embedding_method: str='fasttext') -> None:
    """Generate the files necessary for training and testing

    Args:
        num_tables (int): number of tables to generate
        num_samples (int): number of triples to generate
        outdir (str, optional): directory where to save the generated data. Defaults to "/home/francesco.pugnaloni/wikipedia_tables/small_dataset_debug".
        w (int, optional): max width of the datasets. Defaults to 10.
        h (int, optional): max height of the datasets. Defaults to 10.
        embedding_method (str, optional): embedding method to use, possible 'fasttext' or 'BERT'. Defaults to 'fasttext'.
    """
    random.seed(42)
    dl = generateDatasetList(num_tables, h, w)
    td = {str(i):dl[i] for i in range(len(dl))}
    with open(outdir+'/tables.pkl', 'wb') as f:
        pickle.dump(td, f)
    dg = generate_graph_dictionary(outdir+'/tables.pkl', outdir+'/graphs.pkl', embedding_generation_method=embedding_method)
    triples = []
    for _ in range(num_samples):
        a = random.randint(0,len(dg)-1)
        b = random.randint(0,len(dg)-1)
        o = random.random()
        triples.append([a,b,o])
    pd.DataFrame(triples).to_csv(outdir+"/triples.csv", index=False)

def load_test_training_stuff(filedir: str="/home/francesco.pugnaloni/wikipedia_tables/small_dataset_debug") -> dict:
    """Function to load the training and testing stuff

    Args:
        filedir (str, optional): path to the directory containing the files. Defaults to "/home/francesco.pugnaloni/wikipedia_tables/small_dataset_debug".

    Returns:
        dict: dictionary containing ('tables', 'graphs', 'triples')
    """
    with open(filedir+'/tables.pkl', 'rb') as f:
        td = pickle.load(f)
    with open(filedir+'/graphs.pkl', 'rb') as f:
        gd = pickle.load(f)
    triples = pd.read_csv(filedir+'/triples.csv')

    return {'tables':td, 'graphs':gd, 'triples':triples}

if __name__ == '__main__':
    #generate_test_training_stuff(30, 100, outdir="/home/francesco.pugnaloni/GNNTE/tmp", embedding_method='BERT')
    extract_token_list_to_populate_tables("/home/francesco.pugnaloni/GNNTE/wikipedia_datasets/1MR/full_table_dict_with_id.pkl","/home/francesco.pugnaloni/GNNTE/synthetic_dataset/tokenlist1M.pkl")
    #extract_token_list_to_populate_tables("/home/francesco.pugnaloni/GNNTE/wikipedia_datasets/1kR/1kR.pkl","/home/francesco.pugnaloni/GNNTE/synthetic_dataset/tokenlist50k.pkl")