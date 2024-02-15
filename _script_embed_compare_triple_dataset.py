from _script_embed_table_collection import *

class EmbeddingsTriplesDataset(Dataset):
    
    def __init__(self, triples: pd.DataFrame, embedding_dict: dict) -> None:
        """
        The init method

        Args:
            triples (pd.DataFrame): triple file containing the samples
            embedding_dict (dict): dictionary containing the computed table embeddings
        """
        super(EmbeddingsTriplesDataset, self).__init__()
        self.triples = triples
        self.embeddings = embedding_dict

    def len(self) -> int:
        """len method

        Returns:
            int: number of samples in the dictionary
        """
        return len(self.triples)
    
    def get(self, idx:int) -> tuple:
        """get method

        Args:
            idx (int): index of a sample

        Returns:
            tuple: triple containing 2 embeddings and their overlap ratio
        """
        t = self.triples.iloc[idx][:]
        try:
            g1 = self.embeddings[str(t.iloc[0])]
            g2 = self.embeddings[str(t.iloc[1])]
        except:
            g1 = self.embeddings[str(int(t.iloc[0]))]
            g2 = self.embeddings[str(int(t.iloc[1]))]
        return g1, g2, t.iloc[2]
    
class TriplesDataset(Dataset):
    
    def __init__(self, triples: pd.DataFrame) -> None:
        """
        The init method

        Args:
            triples (pd.DataFrame): triple file containing the samples
            embedding_dict (dict): dictionary containing the computed table embeddings
        """
        super(TriplesDataset, self).__init__()
        self.triples = triples

    def len(self) -> int:
        """len method

        Returns:
            int: number of samples in the dictionary
        """
        return len(self.triples)
    
    def get(self, idx:int) -> tuple:
        """get method

        Args:
            idx (int): index of a sample

        Returns:
            tuple: triple containing 2 embeddings and their overlap ratio
        """
        t = self.triples.iloc[idx][:]
        try:
            g1 = str(t.iloc[0])
            g2 = str(t.iloc[1])
        except:
            g1 = str(int(t.iloc[0]))
            g2 = str(int(t.iloc[1]))
        return g1, g2, t.iloc[2]

def compute_embeddings(model_file: str, graph_dict: dict, batch_size: int=128, mode: str='batch') -> dict:
    """method to compute the embeddings of the graphs in a graph_dictionary

    Args:
        model_file (str): path to a model checkpoint
        graph_dict (dict): instance of graph dictionary
        batch_size (int, optional): size of the batches if using batch mode. Defaults to 128.
        mode (str, optional): mode of usage, 'batch' or 'sequential' are accepted. Defaults to 'batch'.

    Returns:
        dict: a dicitonary that associates the tabels to their embeddings
    """
    model = GNNTE(model_file=model_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gd = GraphsDataset(graph_dict)
    
    dataloader = DataLoader(gd, 
                        batch_size=batch_size,  
                        num_workers=0, 
                        shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings = embed(model, dataloader, device)

    index_to_table = gd.keys

    embeddings_dict = {index_to_table[i]:embeddings[i] for i in range(len(index_to_table))}

    return embeddings_dict

def  compute_overlaps(triples: pd.DataFrame, embedding_dict: dict, mode: str='batch', batch_size: int=128) -> dict:
    """method to compute overlaps between tables

    Args:
        triples (pd.DataFrame): triple dataset
        embedding_dict (dict): dictionary containing the embeddings
        mode (str, optional): mode of usage, accepted 'batch' and 'sequential'. Defaults to 'batch'.
        batch_size (int, optional): size of the training batches. Defaults to 128.

    Returns:
        dict: dictionary containing stats about the execution
    """
    out_predictions = {
        'l_id' : [],
        'r_id' : [],
        'overlap_pred' : [],
        'overlap_true' : [],
        'AE' : []
    }

    gd = TriplesDataset(triples)
    dataloader = DataLoader(gd, 
                        batch_size=batch_size,  
                        num_workers=0, 
                        shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'sequential':
        batch_size = 1
    y_pred = None
    y_true = None
    with torch.no_grad():
        for batch in dataloader:
            # to device
            l_id = batch[0]
            r_id = batch[1]
            emb_l = embedding_dict[l_id].to(device)
            emb_r = embedding_dict[r_id].to(device)

            logits = F.cosine_similarity(emb_l, emb_r, dim=1)
            logits[logits < 0] = 0
            
            y = batch[2]
                # Save the predictions and the labels
            if y_pred is None:
                y_pred = logits  
                y_true = batch[2]
            else:
                y_pred = torch.cat((y_pred, logits))
                y_true = torch.cat((y_true, y))
            out_predictions['l_id'].append(l_id)
            out_predictions['r_id'].append(r_id)
            out_predictions['overlap_pred'].append(float(logits.cpu()))
            out_predictions['overlap_true'].append(y)
            out_predictions['AE'].append(abs(float(logits.cpu())-y))

    out = {'y_pred':y_pred, 'y_true':y_true}
    out.update(compute_metrics(y_pred, y_true))
    out['result_table'] = pd.DataFrame(out_predictions)

    return out
    
    

def run_experiment(model_file: str, triple_dataset_file: str, 
                   table_dict_path: str=None, graph_dict_path: str=None, embeddings_dict_path: str=None, 
                   embedding_mode: str='batch', experiment_data_file_path: str=None,
                   result_table_out_path: str=None) -> dict:
    """Function to run a full experiment pipeline

    Args:
        model_file (str): path to a model checkpoint
        triple_dataset_file (str): path to a dataset containing the triples (t1, t2, label)
        table_dict_path (str, optional): path to a table dicitonary file saved in memory. Defaults to None.
        graph_dict_path (str, optional): path to a graph dicitonary file saved in memory. Defaults to None.
        embeddings_dict_path (str, optional): path to an embedding dicitonary file saved in memory. Defaults to None.
        embedding_mode (str, optional): embedding approach, accepted 'batch' or 'sequential'. Defaults to 'batch'.
        experiment_data_file_path (str, optional): file where to save the output. Defaults to None.
        result_table_out_path (str, optional): path where to save the result table with AE and predicted overlap. Defaults to None

    Raises:
        Exception: unsupported inputs

    Returns:
        dict: dict containing stats about the experiment
    """
    start_global = time.time()
    start = time.time()
    performances = {}
    #Graphs construction
    if (graph_dict_path == None) and (table_dict_path != None) :
        graph_dict = generate_graph_dictionary(table_dict_path, out_path=None, save_graph_dict=False)
        end = time.time()
        print(f'Graphs constructed in: {(end-start)*1000}ms\n')
        performances['graph_constr'] = (end-start)*1000
    elif embeddings_dict_path == None:
        with open(graph_dict_path,'rb') as f:
            graph_dict = pickle.load(f)
        end = time.time()
        print(f'Graphs constructed in: {(end-start)*1000}ms\n')
        performances['graph_constr'] = (end-start)*1000
    

    start = time.time()
    #Embeddings computation
    print('Emebdding generation starts')
    if embeddings_dict_path:
        with open(embeddings_dict_path,'rb') as f:
            embedding_dict = pickle.load(f)
    else:
        embedding_dict = compute_embeddings(model_file=model_file, graph_dict=graph_dict)
    end = time.time()
    print(f'Embeddings generated in: {(end-start)*1000}ms\n')
    performances['emb_gen'] = (end-start)*1000
    
    #Overlap ratios computation
    start = time.time()
    triples = pd.read_csv(triple_dataset_file)
    if embedding_mode == 'batch':
        results = compute_overlaps(triples, embedding_dict, 'batch')
    elif embedding_mode == 'sequential':
        results = compute_overlaps(triples, embedding_dict, 'sequential')
    else:
        raise Exception
    end = time.time()
    print(f'Overlaps computed in: {(end-start)*1000}ms\n')
    performances['overlap_comp'] = (end-start)*1000
    print(f'Total t_exec: {(end-start_global)*1000}ms\n')
    performances['total_texec'] = (end-start_global)*1000
    #print(f'Results: \n{results}')
    performances.update(results)
    
    #Saving performances
    with open(experiment_data_file_path, "wb") as f1:
        pickle.dump(performances, f1)
    print(f'Performancesc: {performances}')

    if result_table_out_path:
        performances['result_table'].to_csv(result_table_out_path, index=False)

    return results

if __name__ == "__main__":
    results = run_experiment(model_file="/home/francesco.pugnaloni/GNNTE/models/GNNTE_1M_thesis.pth",
                            #table_dict_path="/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/1M_wikitables_disjointed/table_dict_full.pkl",
                            triple_dataset_file='/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_samples.csv',
                            embeddings_dict_path='/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1MR/embeddings_full_wikidata_GNNTE_1M_thesis.pkl',
                            #graph_dict_path='/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1MR/graphs.pkl'
                            result_table_out_path='/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_samples.csv/test_results_plottable.csv'
                            )
    