from _script_embed_compare_triple_dataset import *
import tqdm
import matplotlib.pyplot as plt

def run_experiment_instance(model_file: str, table_dict: dict) -> dict:
    """Given a table dict this function computes some performance measures about the speed of the framework in the embedding generation

    Args:
        model_file (str): path to a model checkpoint of GNNTE
        table_dict_path (dict): instanced table_dict

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

        embeddings = embed(model, dataloader, device)

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

    return experiment_data

def update_table_dict(table_dict: dict, experiment_data: dict) -> dict:
    outliers = {}
    for k in table_dict.keys():
        if experiment_data[k]['t_tot'] > 1000:
            outliers[k] = table_dict[k]
    return outliers

def run_experiment(model_file: str, table_dict_path: str | dict, experiment_data_file_path: str=None, iters: int=5) -> dict:
    print('Loading table_dict....')
    if type(table_dict_path) is dict:
        table_dict = table_dict_path
    else:
        with open(table_dict_path, 'rb') as f:
            table_dict = pickle.load(f)

    print('table_dict loaded')
    experiment_data = {}
    for _ in range(iters):
        if len(experiment_data.values()) != 0:
            table_dict = update_table_dict(table_dict, experiment_data)
            new_exp_data = run_experiment_instance(model_file=model_file, table_dict=table_dict)
            for k in new_exp_data.keys():
                if new_exp_data[k]['t_tot'] < experiment_data[k]['t_tot']:
                    experiment_data[k] = new_exp_data[k]
        else:
            experiment_data = run_experiment_instance(model_file=model_file, table_dict=table_dict)

    if experiment_data_file_path:
        with open(experiment_data_file_path, 'wb') as f:
            pickle.dump(experiment_data,f)

def visualize_scatter_plot(exp_data_file: str) -> None:
    with open(exp_data_file, 'rb') as f:
        data = pickle.load(f)

    keys = list(data.keys())

    # areas = [data[k]['area'] for k in keys if data[k]['area']!=0]
    # t_execs = [data[k]['t_tot'] for k in keys if data[k]['area']!=0]

    areas = [data[k]['area'] for k in keys]
    t_execs = [data[k]['t_tot'] for k in keys]

    # Create a scatter plot
    plt.scatter(areas, t_execs, s=50)

    # Add labels and title
    plt.xlabel('Table_area')
    plt.ylabel('Embedding_generation_time (ms)')
    plt.title(exp_data_file)


if __name__ == '__main__':
    run_experiment(model_file='/home/francesco.pugnaloni/GNNTE/models/GNNTE_1M_thesis.pth', 
                   #table_dict_path="/home/francesco.pugnaloni/GNNTE/Datasets/gittables_datasets/gittables_full.pkl",
                   #table_dict_path='/home/francesco.pugnaloni/GNNTE/Datasets/just_1k_tables.pkl',
                   table_dict_path='/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/git_wiki_joined/git_wiki_joined.pkl',
                   #experiment_data_file_path="/home/francesco.pugnaloni/GNNTE/run_data/gen_emb_seq/emb_speed_gittables_800k.pkl")
                   #experiment_data_file_path="/home/francesco.pugnaloni/GNNTE/Datasets/just_1k_tables_stats.pkl")
                   experiment_data_file_path="/home/francesco.pugnaloni/GNNTE/run_data/gen_emb_seq/emb_speed_git_wiki_full_optimezed_5_iters.pkl")
