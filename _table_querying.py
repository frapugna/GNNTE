from GNNTE import *
from graph import *
from _script_embed_table_collection import *
from sklearn.metrics import ndcg_score
from data_visualization import *

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

def compute_model_stats(batch: pd.DataFrame, prediction_label: str, t_exec_label: str) -> dict:
    """for every query compute total time, and sort the predicitons 

    Args:
        batch (pd.DataFrame): subset containing data for a single querying
        prediction_label (str): label with the prediction
        t_exec_label (str): label with the execution time for overlap computation

    Returns:
        dict: dictionary containing the data
    """
    out = {}
    out['n_nans'] = batch[prediction_label].isna().sum()
    out['overlap_computation_time'] = batch[t_exec_label].sum()
    overlaps = [(batch.iloc[r]['s_id'], batch.iloc[r][prediction_label]) for r in range(batch.shape[0])]
    start = time.time()
    overlaps = sorted(overlaps, key=lambda tup: tup[1], reverse=True)
    end = time.time()
    out['overlaps'] = overlaps
    out['total_time'] = out['overlap_computation_time'] + (end-start)
    return out

def prepare_plot_data_with_solth_baseline_armadillo(sloth_baseline_out: str | pd.DataFrame, armadillo_out: str | dict, query_set: str | set, outpath: str) -> dict:
    """Given a table querying file from sloth generate a dictionary that merges its data with armadillo

    Args:
        sloth_baseline_out (str | pd.DataFrame): path to csv containing sloth's output
        armadillo_out (str | dict): path to a pickle containing armadillo's output
        query_set (str | set): path to a pickle containing the query set
        outpath (str): path where to save the restults

    Returns:
        dict: results
    """
    print('Loading sloth data')
    if isinstance(sloth_baseline_out, str):
        sloth_baseline_out = pd.read_csv(sloth_baseline_out)
    print('Loading armadillo data')
    if isinstance(armadillo_out, str):
        with open(armadillo_out, 'rb') as f:
            armadillo_out = pickle.load(f)
    print('Loading query set data')
    if isinstance(query_set, str):
        with open(query_set, 'rb') as f:
            query_set = pickle.load(f)
    
    d_na = sloth_baseline_out[sloth_baseline_out['a%'].isna()]
    print('Starting removing armadillo\'s overtimes')
    dropped_elements = 0
    for k in tqdm(query_set):
        to_drop = set(d_na[d_na['r_id'] == k]['s_id'])
        if len(to_drop) == 0:
            continue
        tmp = armadillo_out[k]['overlaps']
        new_overlaps = [tmp[i] for i in range(len(tmp)) if tmp[i][0] not in to_drop]
        armadillo_out[k]['overlaps'] = new_overlaps
        dropped_elements = dropped_elements + (len(tmp)-len(armadillo_out[k]['overlaps']))
    print(f'Dropped {dropped_elements} elements from armadillo\'s output')
    old_len_sloth = len(sloth_baseline_out)
    sloth_baseline_out = sloth_baseline_out.dropna(subset=['a%'])
    print(f'Dropped {old_len_sloth - len(sloth_baseline_out)} elements from slot\'s output')
    out = {
        'sloth':{},
        'jsim':{},
        'overlap_set_sim':{},
        'armadillo':{}
    }
    print('Stats computation starts')
    for k in tqdm(query_set):
        batch = sloth_baseline_out[sloth_baseline_out['r_id'] == k]
        out['sloth'][k] = compute_model_stats(batch, 'a%', 'total_time')
        out['jsim'][k] = compute_model_stats(batch, 'jsim','jsim_time')
        out['overlap_set_sim'][k] = compute_model_stats(batch, 'josie', 'josie_time')
        out['armadillo'][k] = armadillo_out[k]
    with open(outpath, 'wb') as f:
        pickle.dump(out, f)
    return out

def compute_precision_at_k(true: set, predicted: set) -> float:
    return len(true & predicted) / len(true)

def compute_mean_precision_at_k(predictions_dict: dict, k) -> tuple:
    jsim_precisions = []
    overlap_set_sim_precisions = []
    armadillo_precisions = []
    for i in predictions_dict.keys():
        sloth = set(predictions_dict[i]['sloth'][:k])
        jsim = set(predictions_dict[i]['jsim'][:k])
        overlap_set_sim = set(predictions_dict[i]['overlap_set_sim'][:k])
        armadillo = set(predictions_dict[i]['armadillo'][:k])
        jsim_precisions.append(compute_precision_at_k(sloth, jsim))
        overlap_set_sim_precisions.append(compute_precision_at_k(sloth, overlap_set_sim))
        armadillo_precisions.append(compute_precision_at_k(sloth, armadillo))
    return sum(jsim_precisions)/len(jsim_precisions), sum(overlap_set_sim_precisions)/len(overlap_set_sim_precisions), sum(armadillo_precisions)/len(armadillo_precisions)

def build_data_for_precision_at_k(predictions_dict: dict, query_set: set) -> dict:
    out = {}
    for k in tqdm(query_set):
        out[k] = {
            'sloth':[t[0] for t in predictions_dict['sloth'][k]['overlaps']],
            'jsim':[t[0] for t in predictions_dict['jsim'][k]['overlaps']],
            'overlap_set_sim':[t[0] for t in predictions_dict['overlap_set_sim'][k]['overlaps']],
            'armadillo':[t[0] for t in predictions_dict['armadillo'][k]['overlaps']]
        }
    return out

def prepare_data_frame_precision_at_k(predictions_dict: str | dict, query_set: set | str, outpath: str, max_k: int=100) -> pd.DataFrame:
    if isinstance(predictions_dict, str):
        with open(predictions_dict, 'rb') as f:
            predictions_dict = pickle.load(f)
    if isinstance(query_set, str):
        with open(query_set, 'rb') as f:
            query_set = pickle.load(f)

    print('Building data for precision at k')
    predictions_dict = build_data_for_precision_at_k(predictions_dict, query_set)
    out = {
        'k':[],
        'precision_at_k_jsim':[],
        'precision_at_k_overlap_set_sim':[],
        'precision_at_k_armadillo':[]
    }
    for k in tqdm(range(1, max_k+1)):
        jsim, oset, arm = compute_mean_precision_at_k(predictions_dict, k)
        out['k'].append(k)
        out['precision_at_k_jsim'].append(jsim)
        out['precision_at_k_overlap_set_sim'].append(oset)
        out['precision_at_k_armadillo'].append(arm)
    out = pd.DataFrame(out)
    out.to_csv(outpath, index=False)
    return out

def add_armadillo_predictions_to_sloth_file(sloth_baseline_out: str | pd.DataFrame, armadillo_out: dict | str, outpath: str) -> None:
    if isinstance(sloth_baseline_out, str):
        sloth_baseline_out = pd.read_csv(sloth_baseline_out)
    if isinstance(armadillo_out, str):
        with open(armadillo_out, 'rb') as f:
            armadillo_out = pickle.load(f)
    arm_preds = {}
    for k in armadillo_out.keys():
        arm_preds[k] = {}
        for t in armadillo_out[k]['overlaps']:
            arm_preds[k][t[0]] = t[1]
    sloth_baseline_out.dropna(inplace=True)
    arma = []
    for r in tqdm(range(sloth_baseline_out.shape[0])):
        arma.append(arm_preds[sloth_baseline_out.iloc[r]['r_id']][sloth_baseline_out.iloc[r]['s_id']])
    
    sloth_baseline_out['armadillo'] = arma
    sloth_baseline_out.to_csv(outpath, index=False)

def compute_ndcg_at_k(table_querying_arm_sloth: str | pd.DataFrame, query_set: str | set, outpath: str, k_max: int=100) -> pd.DataFrame:
    if isinstance(table_querying_arm_sloth, str):
        table_querying_arm_sloth = pd.read_csv(table_querying_arm_sloth)
    if isinstance(query_set, str):
        with open(query_set, 'rb') as f:
            query_set = pickle.load(f)
    
    out = {
        'k':[],
        'query_table':[],
        'armadillo':[],
        'overlap_set_sim':[],
        'jsim':[]
    }

    for k in range(0, k_max+1, 10):
        if k == 0:
            k=1
        print(f'Current k: {k}')
        for t in tqdm(query_set):
            out['k'].append(k)
            out['query_table'].append(t)
            curr_subset = table_querying_arm_sloth[table_querying_arm_sloth['r_id']==t]
            out['armadillo'].append(ndcg_score(y_true=np.array(curr_subset[['a%']]).reshape(1,-1), y_score=np.array(curr_subset[['armadillo']]).reshape(1,-1), k=k))
            out['jsim'].append(ndcg_score(y_true=np.array(curr_subset[['a%']]).reshape(1,-1), y_score=np.array(curr_subset[['jsim']]).reshape(1,-1), k=k))
            out['overlap_set_sim'].append(ndcg_score(y_true=np.array(curr_subset[['a%']]).reshape(1,-1), y_score=np.array(curr_subset[['josie']]).reshape(1,-1), k=k))

    k = 10_000
    for t in query_set:
        out['k'].append(k)
        out['query_table'].append(t)
        curr_subset = table_querying_arm_sloth[table_querying_arm_sloth['r_id']==t]
        out['armadillo'].append(ndcg_score(y_true=np.array(curr_subset[['a%']]).reshape(1,-1), y_score=np.array(curr_subset[['armadillo']]).reshape(1,-1), k=k))
        out['jsim'].append(ndcg_score(y_true=np.array(curr_subset[['a%']]).reshape(1,-1), y_score=np.array(curr_subset[['jsim']]).reshape(1,-1), k=k))
        out['overlap_set_sim'].append(ndcg_score(y_true=np.array(curr_subset[['a%']]).reshape(1,-1), y_score=np.array(curr_subset[['josie']]).reshape(1,-1), k=k))
    out = pd.DataFrame(out)
    out.to_csv(outpath, index=False)
    return out

def compare_models_ndcg(data: pd.DataFrame | str, bin_criterion: str='k', bins_name: str='Correct Label', out_pdf: str=None, font_scale: float=0.7) -> pd.DataFrame:
    """Function to plot an histogram to compare performances of different models depending on their range of error 

    Args:
        data (pd.DataFrame | str): data frame containing the results
        bin_criterion (str, optional): parameter to generate the 10 bins, must be with values in [0,1]. Defaults to 'a%'.
        bins_name (str, optional): name of the bins. Defaults to 'AE'.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
    new_data = {
        'k':[],
        'Approach':[],
        'NDCG_Score':[]
    }
    for k in [1, 10, 20, 30, 40, 50, 60, 70,80,90,100,10_000]:
        t = data[data[bin_criterion] == k]
        if k == 10_000:
            k = '10k'
        new_data['Approach'].append('Armadillo')
        new_data['k'].append(k)
        new_data['NDCG_Score'].append(round(np.mean(t['armadillo']),2))

        new_data['Approach'].append('Overlap Set Similarity')
        new_data['k'].append(k)
        new_data['NDCG_Score'].append(round(np.mean(t['overlap_set_sim']),2))
        
        new_data['Approach'].append('Jaccard Similarity')
        new_data['k'].append(k)
        new_data['NDCG_Score'].append(round(np.mean(t['jsim']),2))
        
    
    df = pd.DataFrame(new_data)
    sns.set_theme(font_scale=font_scale, style="whitegrid")
    sns.barplot(data=df, x='k', y='NDCG_Score', hue='Approach')

    if isinstance(out_pdf, str):
        plt.tight_layout()
        plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    return df


if __name__ == '__main__':
    # run_table_querying_experiment(query_set='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/query_set_1k.pkl',
    #                               target_set='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/data_lake_10k.pkl',
    #                               model='/home/francesco.pugnaloni/GNNTE/model_no_perfect_matches_gittables.pth',
    #                               target_embedding_dict='/home/francesco.pugnaloni/GNNTE/tmp/em_git_model_trained_without_perfect_matches.pkl',
    #                               table_dict='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_dict_796970_good.pkl',
    #                               outpath='/home/francesco.pugnaloni/GNNTE/evaluation/table_querying_results_armadillo_no_csv.pkl')

    # prepare_plot_data_with_solth_baseline_armadillo(
    #     sloth_baseline_out='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/table_querying_tmp_data/sloth_predictions.csv',
    #     armadillo_out='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/table_querying_tmp_data/table_querying_results_armadillo_no_csv.pkl',
    #     query_set='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/query_set_1k.pkl',
    #     outpath='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/table_querying_tmp_data/all_predictions_dict.pkl'
    # )
    # prepare_data_frame_precision_at_k(predictions_dict='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/table_querying_tmp_data/all_predictions_dict.pkl', 
    #                                   query_set='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/query_set_1k.pkl',
    #                                   outpath='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/data_plot_precision_at_k.csv')
    # add_armadillo_predictions_to_sloth_file(sloth_baseline_out='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/table_querying_tmp_data/sloth_predictions.csv', 
    #                                         armadillo_out='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/table_querying_tmp_data/table_querying_results_armadillo_no_csv.pkl', 
    #                                         outpath='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/sloth_armadillo_predictions_table_querying.csv')
    compute_ndcg_at_k('/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/sloth_armadillo_predictions_table_querying.csv', '/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/query_set_1k.pkl',
                      '/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_querying/ndcg_data.csv')