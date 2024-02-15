import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import torch.nn.functional as F

def clean_sloth(path: str, outpath: str) -> None:
    df = pd.read_csv(path)
    out = df[['r_id', 's_id', 'a%']]
    out = out.fillna(0)
    out.to_csv(outpath, index=False)

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

if __name__ == '__main__':
    clean_sloth('/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/labelled/old_data/test_stats.csv','/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/labelled/old_data/test_stats_cleaned.csv')
    # re_evaluate_sloth_out(cleaned_sloth_output='/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/gittables_labelled_no_rep_train_test/train_stats_cleaned.csv',
    #                       embedding_dict='/home/francesco.pugnaloni/GNNTE/Datasets/gittables_datasets/embeddings_gittables_model_wikidata_450k_GraphSAGE_50ep.pkl',
    #                       out_path='/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/gittables_labelled_no_rep_train_test/results/train_with_predicted_and_AE_wiki_450k.csv'
    #                       )