from graph import *
import tqdm
import sys
import time

def generate_graph_dictionary(table_dict_path: str | dict, out_path: str, embedding_generation_method: str='sha256', save_graph_dict: bool=True) -> dict:
    """Generate a graph dictionary from a table dictionary

    Args:
        table_dict_path (str): path to the table dictionary
        out_path (str): path to the file where to save the new graph dictionary
        embedding_generation_method (str, optional): approach used to generate embeddings, possible values are 'fasttext' and 'BERT'. Defaults to 'fasttext'
        save_graph_dict (bool, optional): if true the graph_dict will be dumped in the out_path. Deafaults to true
    Returns:
        dict: the generated graph dictionary
    """
    start = time.time()
    print('Loading table_dict.....')
    if isinstance(table_dict_path, str):
        try:
            with open(table_dict_path,'rb') as f:
                table_dict = pickle.load(f)
        except:
            raise Exception("table_dict not found")
    else:
        table_dict = table_dict_path
    print('Table dict loaded')
    end = time.time()
    print(f'Table loaded in: {(end-start)}s\n')

    start_interm = time.time()
    print('Istantiating embedding buffer.....')
    if embedding_generation_method == 'fasttext':
        embedding_buffer = FasttextEmbeddingBuffer(model='fasttext-wiki-news-subwords-300')
        print('Instantiating String_token_preprocessor.....')
        string_token_preprocessor = String_token_preprocessor()
        print('String_token_preprocessor instantiated\n')
    elif embedding_generation_method == 'BERT':
        embedding_buffer = Bert_Embedding_Buffer()
    elif embedding_generation_method == 'sha256':
        embedding_buffer = None
    else:
        print('Embedding generation method not accepted, try "fasttext", "BERT", or "sha256"')
        raise NotImplementedError()
    print('Embedding buffer instantiated')
    end = time.time()
    print(f'Embedding_buffer instantiated in: {(end-start_interm)}s\n')

    
    out = {}

    start_interm = time.time()
    print('Graphs generation starts.....')
    for k in tqdm.tqdm(table_dict.keys()):
        try:
            if embedding_generation_method == 'sha256':
                out[k] = Graph(table_dict[k], k, embedding_buffer_type=embedding_generation_method, merge_nodes_same_value=False)
            else:
                out[k] = Graph(table_dict[k], k, embedding_buffer_type=embedding_generation_method, embedding_buffer=embedding_buffer, string_preprocess_operations=string_token_preprocessor, token_length_limit=1000)
        except:
            out[k] = None
    print('Graph generation ends')
    if save_graph_dict:
        print('Saving output')
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)   
        print('Output saved')
    end = time.time()
    print(f'Graph_dict generated in: {(end-start_interm)}s')
    print(f'Total t_exec: {(end-start)}s')
    return out

if __name__ == "__main__":

    # n_params = len(sys.argv) - 1
    # expected_params = 3
    # if n_params != expected_params:
    #     raise ValueError(f'Wrong number of parameters, you provided {n_params} but {expected_params} are expected. \nUsage is: {sys.argv[0]} table_dict_path out_directory_path embedding_generation_method')
    # table_dict_path = sys.argv[1]
    # out_directory_path = sys.argv[2]
    # embedding_generation_method = sys.argv[3]

    table_dict_path = '/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/full_table_dict_with_id.pkl'
    out_directory_path = '/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/1M_wikitables_disjointed/graphs_sha256_null_not_0_no_merge_nodes.pkl'
    embedding_generation_method = 'sha256'

    graph_dict = generate_graph_dictionary(table_dict_path, out_directory_path, embedding_generation_method)
