from _script_embed_compare_triple_dataset import *
import tqdm
import matplotlib.pyplot as plt

def run_experiment_instance(model_file: str, table_dict: dict, embeddings: dict=None) -> dict:
    """Given a table dict this function computes some performance measures about the speed of the framework in the embedding generation

    Args:
        model_file (str): path to a model checkpoint of GNNTE
        table_dict_path (dict): instanced table_dict
        embeddings (dict): dictionary containing the computed embeddings
    Returns:
        dict: dictionary containing the results
    """
    print('Loading model....')
    model = GNNTE(model_file=model_file)
    print('Model loaded')
    
    print('Loading embedding_buffer....')
    embedding_buffer = FasttextEmbeddingBuffer(model='fasttext-wiki-news-subwords-300')
    print('embedding_buffer loaded....')

    print('Loading string_token_preprocessor....')
    string_token_preprocessor = String_token_preprocessor()
    print('string_token_preprocessor loaded')
    
    experiment_data = {}
    
    for k in tqdm.tqdm(table_dict.keys()):
        t = table_dict[k]

        start = time.time()
        #gen graph
        try:
            g = {k:Graph(table_dict[k], k, embedding_buffer, string_token_preprocessor, token_length_limit=None)}
        except:
            continue
        end_graph = time.time()

        #gen emb
        start_emb = time.time()
        gd = GraphsDataset(g)
        dataloader = DataLoader(gd, batch_size=1, num_workers=0, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        embedding = embed(model, dataloader, device)

        end = time.time()
        data = {
            'n_rows' : t.shape[0],
            'n_cols' : t.shape[1],
            'area' : t.shape[0]*t.shape[1],
            't_graph_gen' : (end_graph-start)*1000,
            't_emb_gen' : (end-start_emb)*1000,   
            't_tot' : (end-start)*1000
        }
        experiment_data[k] = data
        if embeddings != None:
            embeddings[k] = embedding
    if embeddings != None:
        return experiment_data, embeddings
    else:
        return experiment_data

def update_table_dict(table_dict: dict, experiment_data: dict) -> dict:
    outliers = {}
    for k in experiment_data.keys():
        try:
            if experiment_data[str(k)]['t_tot'] > 1000:
                outliers[str(k)] = table_dict[str(k)]
        except:
            if experiment_data[k]['t_tot'] > 1000:
                outliers[k] = table_dict[k]
    return outliers

def run_experiment(model_file: str, table_dict_path: str | dict, experiment_data_file_path: str=None, iters: int=1, embedding_file: str=None) -> dict:
    print('Loading table_dict....')
    if type(table_dict_path) is dict:
        table_dict = table_dict_path
    else:
        with open(table_dict_path, 'rb') as f:
            table_dict = pickle.load(f)

    print('table_dict loaded')
    experiment_data = {}
    embeddings = {}
    for _ in range(iters):
        if len(experiment_data.values()) != 0:
            table_dict = update_table_dict(table_dict, experiment_data)
            new_exp_data = run_experiment_instance(model_file=model_file, table_dict=table_dict)
            for k in new_exp_data.keys():
                try:
                    if new_exp_data[str(k)]['t_tot'] < experiment_data[str(k)]['t_tot']:
                        experiment_data[str(k)] = new_exp_data[str(k)]
                except:
                    if new_exp_data[k]['t_tot'] < experiment_data[k]['t_tot']:
                        experiment_data[k] = new_exp_data[k]
        else:
            experiment_data, embeddings = run_experiment_instance(model_file=model_file, table_dict=table_dict, embeddings=embeddings)
            if embedding_file != None:
                with open(embedding_file,'wb') as f:
                    pickle.dump(embeddings, f)

    if experiment_data_file_path:
        with open(experiment_data_file_path, 'wb') as f:
            pickle.dump(experiment_data,f)

if __name__ == '__main__':
    # run_experiment(
    #     #model_file='/home/francesco.pugnaloni/GNNTE/models/model_wikidata_450k_GraphSAGE_50ep.pth', 
        
    #     model_file='/home/francesco.pugnaloni/GNNTE/models/wikidata/model_wikidata_450k_GraphSAGE_50ep.pth', 
    #     #table_dict_path='/home/francesco.pugnaloni/GNNTE/Datasets/just_1k_tables.pkl',
    #     table_dict_path='/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/full_table_dict_with_id.pkl',
    #     #experiment_data_file_path="/home/francesco.pugnaloni/GNNTE/Datasets/just_1k_tables_stats.pkl",
    #     experiment_data_file_path="/home/francesco.pugnaloni/GNNTE/test_data/tmp/tmp.pkl",
    #     embedding_file = '/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/embeddings/emb_wikifull_450k_15-02.pkl'
    # )
    # dd = pd.read_csv('/home/francesco.pugnaloni/GNNTE/test_data/t_exec/end_2_end_overlap_comparison/t_execs_compared_seconds_full_100tokens.csv')
    # with open('/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_dict_796970_good.pkl','rb') as f:
    #     table_dict = pickle.load(f)

    # table_list = []
    # for r in tqdm.tqdm(range(dd.shape[0])):
    #     table_list.append(dd.iloc[r]['r_id'])
    #     table_list.append(dd.iloc[r]['s_id'])
    # table_list = set(table_list)
    # table_dd = {}
    # for k in table_list:
    #     table_dd[k] = table_dict[k]

    with open('/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/full_table_dict_with_id.pkl', 'rb') as f:
        table_dd = pickle.load(f)

    run_experiment(
            #model_file='/home/francesco.pugnaloni/GNNTE/models/model_wikidata_450k_GraphSAGE_50ep.pth', 
            
            model_file='/home/francesco.pugnaloni/GNNTE/models/wikidata/wikidata_06-03-24_GraphSAGE_50_ep_max_1000_tokens.pth', 
            #table_dict_path='/home/francesco.pugnaloni/GNNTE/Datasets/just_1k_tables.pkl',
            table_dict_path=table_dd,
            #experiment_data_file_path="/home/francesco.pugnaloni/GNNTE/Datasets/just_1k_tables_stats.pkl",
            #experiment_data_file_path="/home/francesco.pugnaloni/GNNTE/test_data/tmp/tmp.pkl",
            #embedding_file = '/home/francesco.pugnaloni/GNNTE/test_data/t_exec/end_2_end_overlap_comparison/embeddings_100token_test_gittables.pkl'
            embedding_file='/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/embeddings/emb_wiki_1000_tokens_17_03.pkl'
        )