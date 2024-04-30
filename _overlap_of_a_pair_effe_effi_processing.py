import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import torch.nn.functional as F
from _script_overlap_computation import *
import matplotlib.pyplot as plt

def clean_sloth(path: str | pd.DataFrame, outpath: str=None) -> pd.DataFrame:
    if isinstance(path, str):
        df = pd.read_csv(path)
    else:
        df = path
    #out = df[['r_id', 's_id', 'a%']]
    out = df.drop(df.columns[0], axis=1)
    out = out.fillna(0)
    if outpath:
        out.to_csv(outpath, index=False)
    return out

def re_evaluate_sloth_out(cleaned_sloth_output: str | pd.DataFrame, embedding_dict: str | dict, out_path: str) -> pd.DataFrame:
    print('Loading outputs')
    if type(cleaned_sloth_output) == str:
        d1 = pd.read_csv(cleaned_sloth_output)
    print('Loading embeddings')
    if type(embedding_dict) == str:
        with open(embedding_dict, 'rb') as f:
            em = pickle.load(f)
    l = []
    out = {
        'l_id' : [],
        'r_id' : [],
        'overlap_pred' : [],
        'overlap_true' : [],
        'AE' : []
    }
    
    for i in tqdm(range(d1.shape[0])):
        predictions = max(float(0), F.cosine_similarity(em[d1.iloc[i].iloc[0]], em[d1.iloc[i].iloc[1]], dim=1))
        try:
            predictions = float(predictions.cpu())
        except:
            pass 
        t = float(d1.iloc[i].iloc[2])

        if pd.isnull(t):
            t = 0
        ae = abs(predictions-t)

        l.append(abs(predictions-t))

        out['l_id'].append(d1.iloc[i].iloc[0])
        out['r_id'].append(d1.iloc[i].iloc[1])
        out['overlap_pred'].append(predictions)
        out['overlap_true'].append(t)
        out['AE'].append(ae)

    df_out = pd.DataFrame(out)

    df_out.to_csv(out_path, index=False)
    print('Output saved')
    
    return df_out

def compute_overlap_ratio(model: GNNTE, dataloader: DataLoader, device: str) -> tuple:
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            emb = model(batch.to(device))
            try:
                embeddings = torch.cat((embeddings, emb), dim=0)
            except:
                embeddings = emb
    return embeddings

def build_graphs(t1: pd.DataFrame, k1: str, t2: pd.DataFrame, k2: str, embedding_buffer: Embedding_buffer=None, string_token_preprocessor: String_token_preprocessor=None) -> tuple:
    if string_token_preprocessor == None:
        g1 = Graph_Hashed_Node_Embs(t1, k1)
        g2 = Graph_Hashed_Node_Embs(t2, k2)
    else:
        g1 = Graph(t1, k1, embedding_buffer, string_token_preprocessor, token_length_limit=100)
        g2 = Graph(t2, k2, embedding_buffer, string_token_preprocessor, token_length_limit=100)
    return g1, g2

def add_areas_cols_rows(sloth_outputs_file: pd.DataFrame | str, table_dict: dict | str, output_file: str=None) -> pd.DataFrame:

    if isinstance(sloth_outputs_file, str):
        sloth_outputs_file = pd.read_csv(sloth_outputs_file)
    
    if isinstance(table_dict, str):
        with open(table_dict, 'rb') as f:
            table_dict = pickle.load(f)

    new_cols = {
        'r_rows':[],
        'r_cols':[],
        'r_area':[],
        's_rows':[],
        's_cols':[],
        's_area':[],
        'tot_rows':[],
        'tot_cols':[],
        'tot_area':[]
    }

    for r in tqdm(range(sloth_outputs_file.shape[0])):
        r_table = table_dict[sloth_outputs_file.iloc[r]['r_id']]
        s_table = table_dict[sloth_outputs_file.iloc[r]['s_id']]

        r_rows = r_table.shape[0]
        r_cols = r_table.shape[1]
        r_area = r_rows*r_cols

        s_rows = s_table.shape[0]
        s_cols = s_table.shape[1]
        s_area = s_rows*s_cols

        tot_rows = r_rows + s_rows
        tot_cols = r_cols + s_cols
        tot_area = r_area + s_area

        new_cols['r_rows'].append(r_rows)
        new_cols['r_cols'].append(r_cols)
        new_cols['r_area'].append(r_area)

        new_cols['s_rows'].append(s_rows)
        new_cols['s_cols'].append(s_cols)
        new_cols['s_area'].append(s_area)

        new_cols['tot_rows'].append(tot_rows)
        new_cols['tot_cols'].append(tot_cols)
        new_cols['tot_area'].append(tot_area)
    
    new_cols = pd.DataFrame(new_cols)

    out = pd.concat([sloth_outputs_file, new_cols], axis=1)

    if output_file:
        out.to_csv(output_file, index=False)

    return out

def repeat_test_emb_already_computed(old_file: str | pd.DataFrame, embeddings_dict: str | dict, out_path : str=None) -> pd.DataFrame:
    if isinstance(old_file, str):
        old_file = pd.read_csv(old_file)
    if isinstance(embeddings_dict, str):
        with open(embeddings_dict, 'rb') as f:
            embeddings_dict = pickle.load(f)
    d = {'overlap_computations_repeated_armadillo':[],
         'overlap_computations_no_read_armadillo':[]}
    
    for r in tqdm(range(old_file.shape[0])):
        start = time.time()
        e1 = embeddings_dict[old_file.iloc[r]['r_id']]
        e2 = embeddings_dict[old_file.iloc[r]['s_id']]
        start_no_load = time.time()
        max(float(0), F.cosine_similarity(e1, e2, dim=1))
        end = time.time()

        d['overlap_computations_repeated_armadillo'].append(end-start)
        d['overlap_computations_no_read_armadillo'].append(end-start_no_load)

    
    new_cols = pd.DataFrame(d)
    out = pd.concat([old_file, new_cols], axis=1)

    if out_path:
        out.to_csv(out_path, index=False)

    print(new_cols.describe())

    return out



def recompute_embeddings_overlaps_overlap_computation_time(sloth_outputs_file: str, model_file: str, table_dict: str, output_file: str=None) -> pd.DataFrame:
    """_summary_

    Args:
        sloth_outputs_file (str): file containing the outputs of a sloth run
        model_file (str): path to a trained armadillo model
        table_dict (dict): path to a dictionary containing the tables
        output_file (str, optional): file where to save the enriched dataframe. Defaults to None.

    Returns:
        pd.DataFrame: an erchied dataframe with armadillo execution times
    """
    sloth_data = pd.read_csv(sloth_outputs_file)
    
    print('Loading table dict....')
    with open(table_dict, 'rb') as f:
            table_dict = pickle.load(f)
    print('Table dict loaded')

    print('Loading model....')
    model = GNNTE(model_file=model_file)
    print('Model loaded')

    
    if model.in_channels == 300:
        print('Loading embedding_buffer....')
        embedding_buffer = FasttextEmbeddingBuffer(model='fasttext-wiki-news-subwords-300')
        print('embedding_buffer loaded....')
    else:
        embedding_buffer = None
    
    if model.in_channels == 300:
        print('Loading string_token_preprocessor....')
        string_token_preprocessor = String_token_preprocessor()
        print('string_token_preprocessor loaded')
    else:
        string_token_preprocessor = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exec_times = {
        'graphs_generation_time':[],
        'embeddings_generation_time':[],
        'cos_sim_time_armadillo':[],
        'armadillo_total_time':[],
        'armadillo':[]        
    }

    for r in tqdm(range(sloth_data.shape[0])):
        start = time.time()
        
        #Graphs construction
        start_g = time.time()
        k1 = sloth_data.iloc[r]['r_id']
        k2 = sloth_data.iloc[r]['s_id']
        t1 = table_dict[k1]
        t2 = table_dict[k2]
        g_r, g_s = build_graphs(t1, k1,t2,k2,embedding_buffer,string_token_preprocessor)
        g = {k1:g_r, k2:g_s}
        end_g = time.time()
        exec_times['graphs_generation_time'].append(end_g-start_g)

        #Embeddings generation
        start_e = time.time()
        gd = GraphsDataset(g)
        dataloader = DataLoader(gd, batch_size=2, num_workers=0, shuffle=False)
        embeddings = embed(model, dataloader, device)
        end_e = time.time()
        exec_times['embeddings_generation_time'].append(end_e-start_e)

        #Overlap ratio computation
        start_o = time.time()
        overlap = max(float(0), F.cosine_similarity(embeddings[0], embeddings[1], dim=0))
        end_o = time.time()
        exec_times['cos_sim_time_armadillo'].append(end_o-start_o)

        end = time.time()
        exec_times['armadillo_total_time'].append(end-start)
        
        #try:
        exec_times['armadillo'].append(float(overlap))
        # except:
        #      exec_times['armadillo'].append(float(overlap))
    
    new_cols = pd.DataFrame(exec_times)
    new_cols['AE_armadillo'] = abs(sloth_data['a%'] - new_cols['armadillo'])
    out = pd.concat([sloth_data, new_cols], axis=1)

    #out = add_areas_cols_rows(sloth_outputs_file=out, table_dict=table_dict)

    if output_file:
        out.to_csv(output_file, index=False)
    
    return out

def add_new_column_prediction_armadillo(old_data: str | pd.DataFrame, embedding_dict: str | dict, out_path: str, label: str) -> pd.DataFrame:
    print('Loading outputs')
    if type(old_data) == str:
        d1 = pd.read_csv(old_data)
    print('Loading embeddings')
    if type(embedding_dict) == str:
        with open(embedding_dict, 'rb') as f:
            em = pickle.load(f)
    l = []
    name_pred = f'armadillo_{label}'
    AE_pred = f'armadillo_{label}_AE'
    out = {
        name_pred : [],
        AE_pred : []
    }
    
    for i in tqdm(range(d1.shape[0])):
        predictions = max(float(0), F.cosine_similarity(em[d1.iloc[i].iloc[0]], em[d1.iloc[i].iloc[1]], dim=1))
        try:
            predictions = float(predictions.cpu())
        except:
            pass 
        out[name_pred].append(predictions)

    d1[name_pred] = out[name_pred]
    d1[AE_pred] = abs(d1[name_pred]-d1['a%'])

    d1.to_csv(out_path, index=False)
    print('Output saved')
    
    return d1

if __name__ == '__main__':
    #clean_sloth('/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/labelled/old_data/valid_stats.csv','/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/labelled/old_data/valid_stats_cleaned.csv')
    
    # re_evaluate_sloth_out(cleaned_sloth_output='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/labelled/old_data/valid_stats_cleaned.csv',
    #                       embedding_dict='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/embeddings/embeddings_gittables_model_wikidata_450k_GraphSAGE_50ep.pkl',
    #                       out_path='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/labelled/old_data/valid_stats_cleaned_450k_with_AE.csv'
    #                       )
    recompute_embeddings_overlaps_overlap_computation_time(
        sloth_outputs_file= '/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/evaluation_intermediate/evaluation_test_last.csv',
        model_file='/home/francesco.pugnaloni/GNNTE/best_model_gittables.pth',
        table_dict='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_dict_796970_good.pkl',
        output_file='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/evaluation_intermediate/evaluation_test_last_with_sloth_times.csv'
        )

    # add_areas_cols_rows(sloth_outputs_file='/home/francesco.pugnaloni/GNNTE/test_data/t_exec/end_2_end_overlap_comparison/t_execs_compared_seconds.csv',
    #                     table_dict='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/table_dict_796970_good.pkl',
    #                     output_file='/home/francesco.pugnaloni/GNNTE/test_data/t_exec/end_2_end_overlap_comparison/t_execs_compared_seconds_with_areas.csv'
    #                     )

    repeat_test_emb_already_computed(old_file='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/evaluation_intermediate/evaluation_test_last_with_sloth_times.csv',
                                     embeddings_dict='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/embeddings_best_mae_gittables.pkl',
                                     out_path='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/evaluation_test_last_with_sloth_times_embeddings_retested.csv')
    #add_new_column_prediction_armadillo(old_data='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/baseline_performances/test_set_similarities_gittables_with_armadillo_predictions.csv',
    #                                   embedding_dict='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/embeddings/embeddings_made_with_arm_trained_on_wikidata.pkl',
    #                                   out_path='/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/gittables_effectiveness_last_with_wiki.csv',
    #                                  label='wikitables')
    
    #add_new_column_prediction_armadillo(old_data='/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/test.csv',
    #                                    embedding_dict='/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/embeddings/embeddings_made_with_arm_trained_on_wikitables.pkl',
    #                                    out_path='/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/effectiveness_evaluation.csv',
    #                                    label='wikitables')
    #add_new_column_prediction_armadillo(old_data='/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/effectiveness_evaluation.csv',
    #                                    embedding_dict='/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/embeddings/embeddings_made_with_arm_trained_on_gittables.pkl',
    #                                    out_path='/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/effectiveness_evaluation.csv',
    #                                    label='gittables')