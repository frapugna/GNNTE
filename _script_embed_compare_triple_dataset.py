from _script_embed_table_collection import *

class EmbeddingsTriplesDataset(Dataset):
    
    def __init__(self, triples: pd.DataFrame, embedding_dict: dict) -> None:
        super(EmbeddingsTriplesDataset, self).__init__()
        self.triples = triples
        self.embeddings = embedding_dict

    def len(self) -> int:
        return len(self.triples)
    
    def get(self, idx:int) -> tuple:
        t = self.triples.iloc[idx][:]
        try:
            g1 = self.embeddings[str(t.iloc[0])]
            g2 = self.embeddings[str(t.iloc[1])]
        except:
            g1 = self.embeddings[str(int(t.iloc[0]))]
            g2 = self.embeddings[str(int(t.iloc[1]))]
        return g1, g2, t.iloc[2]
    

def compute_embeddings(model_file: str, graph_dict: dict, batch_size: int=128, mode: str='batch') -> dict:
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

def compute_overlaps(triples: pd.DataFrame, embedding_dict: dict, mode: str='batch', batch_size: int=128) -> dict:
    gd = EmbeddingsTriplesDataset(triples, embedding_dict)
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
            emb_l = batch[0].to(device)
            emb_r = batch[1].to(device)

            logits = F.cosine_similarity(emb_l, emb_r, dim=1)
            y = batch[2]
                # Save the predictions and the labels
            if y_pred is None:
                y_pred = logits  
                y_true = batch[2]
            else:
                y_pred = torch.cat((y_pred, logits))
                y_true = torch.cat((y_true, y))

    out = {'y_pred':y_pred, 'y_true':y_true}
    out.update(compute_metrics(y_pred, y_true))

    return out
    
    

def run_experiment(model_file: str, triple_dataset_file: str, 
                   table_dict_path: str=None, graph_dict_path: str=None, embeddings_dict_path: str=None, 
                   embedding_mode: str='batch', experiment_data_file_path: str=None) -> dict:
    
    #Graphs construction
    if graph_dict_path == None:
        graph_dict = generate_graph_dictionary(table_dict_path, out_path=None, save_graph_dict=False)
    else:
        with open(graph_dict_path,'rb') as f:
            graph_dict = pickle.load(f)

    #Embeddings computation
    if embeddings_dict_path:
        with open(embeddings_dict_path,'rb') as f:
            embedding_dict = pickle.load(f)
    else:
        embedding_dict = compute_embeddings(model_file=model_file, graph_dict=graph_dict)

    #Overlap ratios computation
    triples = pd.read_csv(triple_dataset_file)
    if embedding_mode == 'batch':
        results = compute_overlaps(triples, embedding_dict, 'batch')
    elif embedding_mode == 'sequential':
        results = compute_overlaps(triples, embedding_dict, 'sequential')
    else:
        raise Exception


    #Saving performances


if __name__ == "__main__":

    # n_params = len(sys.argv) - 1
    # expected_params = 3
    # if n_params != expected_params:
    #     raise ValueError(f'Wrong number of parameters, you provided {n_params} but {expected_params} are expected. \nUsage is: {sys.argv[0]} model_file truple_dataset_file table_dict_path')
    # model_file = sys.argv[1]
    # table_dict_path = sys.argv[2]
    # out_directory_path = sys.argv[3]

    results = run_experiment("/home/francesco.pugnaloni/GNNTE/models/GNNTE_1M_thesis.pth", graph_dict_path="/home/francesco.pugnaloni/GNNTE/wikipedia_datasets/1000_samples/graphs.pkl",
                             triple_dataset_file="/home/francesco.pugnaloni/GNNTE/wikipedia_datasets/1000_samples/samples.csv")
    