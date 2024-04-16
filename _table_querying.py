from GNNTE import *
from graph import *
from _script_embed_table_collection import *

def table_querying(t_query: str, target_set: set, model: GNNTE, table_dict: dict, target_embedding_tensor: dict, index_to_table_mapping: dict) -> dict:
    """method to perform a single query operation

    Args:
        t_query (str): name of the table to query the data lake
        target_set (set): set of tables in the data lake
        model (GNNTE): model file to initialize armadillo
        table_dict (dict): dictionary containing all of the tables in the data lake
        target_embedding_tensor (dict): embedding tensor containing all the embeddings of the tables in the data lake
        index_to_table_mapping (dict): mapping to link the elements in the tensor with the table names

    Returns:
        dict: contains: {'total_time', 'overlap_computation_time', 'overlaps'}
    """
    start = time.time()
    g = {t_query:Graph_Hashed_Node_Embs(table_dict[t_query], t_query)}
    gd = GraphsDataset(g)
    dataloader = DataLoader(gd, batch_size=1, num_workers=0, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding = embed(model, dataloader, device)

    overlaps = F.cosine_similarity(embedding, target_embedding_tensor, dim=1)
    end_overlap_comp = time.time()

    overlaps_list = [(index_to_table_mapping[k], float(overlaps[k])) for k in range(overlaps.shape[0])]
    overlaps_list = sorted(overlaps_list, key=lambda tup: tup[1], reverse=True)

    end_sorting_time = time.time()

    out = {'total_time':end_sorting_time-start, 'overlap_computation_time':end_overlap_comp-start, 'overlaps':overlaps_list}
    return out



def build_embedding_tensor(target_set: set, target_embedding_dict: dict) -> set:
    """Given a table dict it provides an unique matrix containing all of the embeddings and a mapping between index and table name

    Args:
        target_set (set): set of the necessary embeddings
        target_embedding_dict (dict): dictionary containing all of the embeddings

    Returns:
        set: set containing (mappings from index to table_name, the embedding tensor)
    """
    mapping = {}
    embedding_list = []
    i = 0
    for k in tqdm(target_set):
        mapping[i] = k
        i+=1
        embedding_list.append(target_embedding_dict[k])
    
    embedding_tensor = torch.vstack(embedding_list)
    return mapping, embedding_tensor


def run_table_querying_experiment(query_set: set | str, target_set: set | str, model: GNNTE | str, target_embedding_dict: dict | str=None, table_dict: dict | str=None, outpath: str=None) -> None:
    """Function to execute the full table querying pipeline

    Args:
        query_set (set | str): set of the tables to use for querying the data lake
        target_set (set | str): data lake to query
        model (GNNTE | str): model to generate the emebddings
        target_embedding_dict (dict | str, optional): embedding dictionary containing the embeddings of the tables inside the data lake. Defaults to None.
        table_dict (dict | str, optional): dictionary containing the tables in the datalake. Defaults to None.
        outpath (str, optional): file where to save the results
    """
    start = time.time()
    print('Loading query set')
    if isinstance(query_set, str):
        with open(query_set, 'rb') as f:
            query_set = pickle.load(f)
    
    print('Loading target set')
    if isinstance(target_set, str):
        with open(target_set, 'rb') as f:
            target_set = pickle.load(f)

    print('Loading model')
    if isinstance(model, str):
        model = GNNTE(model_file=model)

    print('Loading table dict')
    if isinstance(table_dict, str):
        with open(table_dict, 'rb') as f:
            table_dict = pickle.load(f)

    if isinstance(target_embedding_dict, str):
        print('Loading target embedding dict')
        with open(target_embedding_dict, 'rb') as f:
            target_embedding_dict = pickle.load(f)

    print('Building embedding tensor')
    index_to_table_mapping, embedding_tensor = build_embedding_tensor(target_set=target_set, target_embedding_dict=target_embedding_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_tensor = embedding_tensor.to(device)

    start_querying = time.time()
    results = {}
    for t_query in tqdm(query_set):
        results[t_query] = table_querying(t_query=t_query, target_set=target_set, model=model, table_dict=table_dict, target_embedding_tensor=embedding_tensor, index_to_table_mapping=index_to_table_mapping)
    end = time.time()
    total_time = end - start
    query_time = end - start_querying
    print(f'Total time: {total_time}s    Querying time: {query_time}s')
    with open(outpath, 'wb') as f:
        pickle.dump(results, f)
    

if __name__ == '__main__':
    run_table_querying_experiment(query_set='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/query_set_1k.pkl',
                                  target_set='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/data_lake_10k.pkl',
                                  model='/home/francesco.pugnaloni/GNNTE/model_no_perfect_matches_gittables.pth',
                                  target_embedding_dict='/home/francesco.pugnaloni/GNNTE/tmp/em_git_model_trained_without_perfect_matches.pkl',
                                  table_dict='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_dict_796970_good.pkl',
                                  outpath='/home/francesco.pugnaloni/GNNTE/tmp/table_querying_raw.pkl')