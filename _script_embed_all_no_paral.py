from _script_embed_compare_triple_dataset import *
import tqdm

def run_experiment(model_file: str, table_dict_path: str, experiment_data_file_path: str=None) -> dict:
    """Given a table dict this function computes some performance measures about the speed of the framework in the embedding generation

    Args:
        model_file (str): path to a model checkpoint of GNNTE
        table_dict_path (str): path to a table_dict
        experiment_data_file_path (str, optional): file where to save as pickle the results. Defaults to None.

    Returns:
        dict: dictionary containing the results
    """
    print('Loading model....')
    model = GNNTE(model_file=model_file)
    print('Model loaded')

    print('Loading table_dict....')
    with open(table_dict_path, 'rb') as f:
        table_dict = pickle.load(f)
    print('table_dict loaded')
    
    print('Loading embedding_buffer....')
    embedding_buffer = FasttextEmbeddingBuffer(model='fasttext-wiki-news-subwords-300')
    print('embedding_buffer loaded....')

    print('Loading string_token_preprocessor....')
    string_token_preprocessor = String_token_preprocessor()
    print('string_token_preprocessor loaded')
    
    experiment_data = {}
    i=0
    for k in tqdm.tqdm(table_dict.keys()):
        i+=1
        if i > 100:
            break
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

    with open(experiment_data_file_path, 'wb') as f:
        pickle.dump(experiment_data,f)

if __name__ == '__main__':
    run_experiment(model_file='/home/francesco.pugnaloni/GNNTE/models/GNNTE_1M_thesis.pth', 
                   table_dict_path="/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_tables.pkl",
                   experiment_data_file_path="/home/francesco.pugnaloni/GNNTE/tmp/out_stat_e2e_serial.pkl")