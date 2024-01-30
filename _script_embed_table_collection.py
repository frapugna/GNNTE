from GNNTE import *
from _script_generate_graph_dict import generate_graph_dictionary


class TablesDataset(Dataset):
    
    def __init__(self, triples: pd.DataFrame, graphs: dict, tables: Optional[dict] = None) -> None:
        """Init function

        Args:
            triples (pd.DataFrame): Dataframe that contains triples ()'r_id','s_id','table_overlap')
            graphs (dict): a dictionary containing a graph for every key that appears in the triples dataset
            tables (Optional[dict], optional): not implemented. Defaults to None.
        """
        super(GraphTriplesDataset, self).__init__()
        self.triples = triples
        self.graphs = graphs

    def len(self) -> int:
        return len(self.triples)
    
    def get(self, idx:int) -> tuple:
        t = self.triples.iloc[idx][:]
        try:
            g1 = self.graphs[str(t.iloc[0])]
            g2 = self.graphs[str(t.iloc[1])]
        except:
            g1 = self.graphs[str(int(t.iloc[0]))]
            g2 = self.graphs[str(int(t.iloc[1]))]
        return Data(g1.X, g1.edges), Data(g2.X, g2.edges), t.iloc[2]

def embed(model: GNNTE, valid_dataloader: DataLoader, criterion: nn.MSELoss, device: str) -> float:
    avg_loss = 0.0
    model.eval()

    with torch.no_grad():
        for batch in valid_dataloader:
            # to device
            emb_l = model(batch[0].to(device))
            emb_r = model(batch[1].to(device))

            predictions = F.cosine_similarity(emb_l, emb_r, dim=1)

            y = batch[2].to(device)

            # Loss
            loss = criterion(predictions, y)

            avg_loss += loss.item()

    avg_loss = avg_loss / len(valid_dataloader)

    return avg_loss

def generate_table_embeddings(model_file: str, table_dict_path: str=None, out_path: str=None, graph_dict_path: str=None, mode: str='full', save_embeddings: bool=True) -> dict:
    if mode == 'full':
        graph_dict = generate_graph_dictionary(table_dict_path, out_path=None, save_graph_dict=False)
    elif mode == 'embed_graphs':
        with open(graph_dict_path,'rb') as f:
            graph_dict = pickle.load(f)
    else:
        raise NotImplementedError
    model = GNNTE(model_file=model_file)

    eval_loader = DataLoader(GraphTriplesDataset(tmp, graphs), 
                        batch_size=batch_size,  
                        num_workers=num_workers, 
                        shuffle=shuffle)

    
