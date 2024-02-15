from GNNTE import *
from _script_generate_graph_dict import generate_graph_dictionary
from graph import Graph
import sys

class GraphsDataset(Dataset):
    
    def __init__(self, graphs: dict) -> None:
        """init method

        Args:
            graphs (dict): a dictionary containing the graph-versions of the tables to embed
        """
        super(GraphsDataset, self).__init__()
        self.graphs = graphs
        self.keys = list(graphs.keys())

    def len(self) -> int:
        """len method

        Returns:
            int: number of graphs in the dataset
        """
        return len(self.keys)
    
    def get(self, idx:int) -> Graph:
        """get method

        Args:
            idx (int): number from zero to max_length that represents a graph

        Returns:
            Graph: the graph associated with the idx
        """
        k = self.keys[idx]
        try:
            g1 = self.graphs[str(k)]
        except:
            g1 = self.graphs[str(int(k))]
        return Data(g1.X, g1.edges)

def embed(model: GNNTE, dataloader: DataLoader, device: str) -> tuple:
    """_summary_

    Args:
        model (GNNTE): an instance of a GNNTE model
        dataloader (DataLoader): dataloader containing the graph_dataset
        device (str): hardware where to work on

    Returns:
        torch.tensor: tensor with shape number_of_embeddings x embedding_size 
    """
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            emb = model(batch.to(device))
            try:
                embeddings = torch.cat((embeddings, emb), dim=0)
            except:
                embeddings = emb
    return embeddings
    

def generate_table_embeddings(model_file: str, table_dict_path: str=None, out_path: str=None, graph_dict_path: str=None, batch_size: int=1, mode: str='embed_graphs', ) -> dict:
    """Method to embed a collection of pandas dataframes contained in a dictionary

    Args:
        model_file (str): path to an instance of a GNNTE model
        table_dict_path (str, optional): path to the table dictionary, not necessary if mode=='embed_graphs'. Defaults to None.
        out_path (str, optional): path to the file where to save the embedding_file in pickle format, if None nothing is done. Defaults to None.
        graph_dict_path (str, optional): path to the table dictionary, not necessary if mode=='full'. Defaults to None.
        mode (str, optional): mode of operation, accepted 'full' and 'embed_graphs'. Defaults to 'full'.
        batch_size (int, optional): size of the batch in the dataloader. Defaults to 9.

    Raises:
        NotImplementedError: raised if an unsupported mode is required

    Returns:
        dict: dict containing the embeddings associated with the table-names
    """
    if mode == 'full':
        graph_dict = generate_graph_dictionary(table_dict_path, out_path=None, save_graph_dict=False)
    elif mode == 'embed_graphs':
        with open(graph_dict_path,'rb') as f:
            graph_dict = pickle.load(f)
    else:
        raise NotImplementedError
    model = GNNTE(model_file=model_file)
    gd = GraphsDataset(graph_dict)
    dataloader = DataLoader(gd, 
                        batch_size=batch_size,  
                        num_workers=0, 
                        shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    embeddings = embed(model, dataloader, device)

    index_to_table = gd.keys
    
    embeddings_dict = {index_to_table[i]:embeddings[i] for i in range(len(index_to_table))}

    with open(out_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    return embeddings_dict



if __name__ == "__main__":

    # n_params = len(sys.argv) - 1
    # expected_params = 3
    # if n_params != expected_params:
    #     raise ValueError(f'Wrong number of parameters, you provided {n_params} but {expected_params} are expected. \nUsage is: {sys.argv[0]} model_file table_dict_path out_directory_path')
    # model_file = sys.argv[1]
    # table_dict_path = sys.argv[2]
    # out_directory_path = sys.argv[3]

    #table_embeddings = generate_table_embeddings("/home/francesco.pugnaloni/GNNTE/testing_data/GNNTE_fasttext_1k.pth","/home/francesco.pugnaloni/GNNTE/tmp/tables.pkl",out_path='/home/francesco.pugnaloni/GNNTE/testing_data/embeddings.pkl',mode='embed_graphs', graph_dict_path='/home/francesco.pugnaloni/GNNTE/testing_data/graphs.pkl')
    table_embeddings = generate_table_embeddings(
                                    model_file='/home/francesco.pugnaloni/GNNTE/models/GNNTE_1M_thesis.pth',
                                    graph_dict_path='/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1MR/graphs.pkl',
                                    out_path='/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1MR/embeddings_full_wikidata_GNNTE_1M_thesis.pkl',
                                    mode='embed_graphs'
                                    )

    print(f'Generated {len(table_embeddings.values())} embeddings')